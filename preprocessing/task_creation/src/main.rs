use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use num_cpus;
use pcap::Capture;
use pnet::packet::ethernet::EthernetPacket;
use pnet::packet::ip::IpNextHeaderProtocols;
use pnet::packet::ipv4::Ipv4Packet;
use pnet::packet::tcp::TcpPacket;
use pnet::packet::udp::UdpPacket;
use pnet::packet::Packet;
use rand::seq::SliceRandom;
use rand::Rng;
// Add rayon for parallel processing
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{self};
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

#[allow(non_snake_case)]
// Task types
#[derive(Serialize, Deserialize, Debug, Clone)]
enum TaskType {
    BMLM,
    NextBytePrediction,
    WiresharkJSON,
    QuestionAnswering,
    FieldFinding,
}

// Data structures for each task
#[derive(Serialize, Deserialize, Debug, Clone)]
struct BMLMTask {
    task_type: TaskType,
    pcap_bytes: Vec<u8>,
    masked_indices: Vec<usize>,
    masked_values: Vec<u8>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct NextByteTask {
    task_type: TaskType,
    input_bytes: Vec<u8>,
    target_byte: u8,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct JSONTask {
    task_type: TaskType,
    pcap_bytes: Vec<u8>,
    json_output: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct QATask {
    task_type: TaskType,
    question: String,
    pcap_bytes: Vec<u8>,
    answer: String,
}

#[derive(Clone)]
struct PacketData {
    data: Vec<u8>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct FieldFindingTask {
    task_type: TaskType,
    packet_id: usize,
    field_name: String,
    flow_bytes: Vec<u8>,
    field_value: String,
}

// Sample questions for the QA task
const QA_QUESTIONS: [&str; 15] = [
    "What is the source MAC address?",
    "What is the destination MAC address?",
    "What is the EtherType of this frame?",
    "Is this an IPv4 or IPv6 packet?",
    "What is the source IP address?",
    "What is the destination IP address?",
    "What is the IP protocol number?",
    "What is the Time To Live (TTL) value for this IP packet?",
    "What is the DSCP value?",
    "What is the total length of the IP packet?",
    "What is the IP header length?",
    "What is the TCP source port?",
    "What is the TCP destination port?",
    "Is the TCP SYN flag set?",
    "What is the TCP window size?"
];

// Thread-safe file writers wrapper
#[derive(Clone)]
struct ThreadSafeWriters {
    bmlm_writer: Arc<Mutex<BufWriter<File>>>,
    nbp_writer: Arc<Mutex<BufWriter<File>>>,
    json_writer: Arc<Mutex<BufWriter<File>>>,
    qa_writer: Arc<Mutex<BufWriter<File>>>,
    field_writer: Arc<Mutex<BufWriter<File>>>,
}

// Global processor state
struct TaskGenerator {
    writers: ThreadSafeWriters,
    processed_files: Arc<Mutex<HashSet<String>>>,
    output_dir: PathBuf,
}

//
// Helper Functions
//

// Analyze a packet to answer QA question
fn analyze_packet_for_qa(packet: &PacketData, question: &str) -> String {
    if let Some(ethernet) = EthernetPacket::new(&packet.data) {
        match question {
            "What is the source MAC address?" => {
                return ethernet.get_source().to_string();
            }
            "What is the destination MAC address?" => {
                return ethernet.get_destination().to_string();
            }
            "What is the EtherType of this frame?" => {
                return format!("0x{:04x}", u16::from_be(ethernet.get_ethertype().0));
            }
            "Is this an IPv4 or IPv6 packet?" => {
                // Check ethertype for IPv4 (0x0800) or IPv6 (0x86DD)
                let ethertype = u16::from_be(ethernet.get_ethertype().0);
                return if ethertype == 0x0800 {
                    "IPv4".to_string()
                } else if ethertype == 0x86DD {
                    "IPv6".to_string()
                } else {
                    "Neither IPv4 nor IPv6".to_string()
                };
            }
            _ => {}
        }

        // Check for IPv4
        if let Some(ipv4) = Ipv4Packet::new(ethernet.payload()) {
            match question {
                "What is the source IP address?" => {
                    return ipv4.get_source().to_string();
                }
                "What is the destination IP address?" => {
                    return ipv4.get_destination().to_string();
                }
                "What is the IP protocol number?" => {
                    return format!("{}", ipv4.get_next_level_protocol().0);
                }
                "What is the Time To Live (TTL) value for this IP packet?" => {
                    return format!("{}", ipv4.get_ttl());
                }
                // Add more IPv4-specific questions
                _ => {}
            }

            // Check TCP/UDP
            match ipv4.get_next_level_protocol() {
                IpNextHeaderProtocols::Tcp => {
                    if let Some(tcp) = TcpPacket::new(ipv4.payload()) {
                        match question {
                            "Is this packet a TCP segment?" => {
                                return "Yes".to_string();
                            }
                            "What is the TCP source port?" => {
                                return format!("{}", tcp.get_source());
                            }
                            "What is the TCP destination port?" => {
                                return format!("{}", tcp.get_destination());
                            }
                            "Is the TCP SYN flag set?" => {
                                return if tcp.get_flags() & 0x02 != 0 {
                                    "Yes".to_string()
                                } else {
                                    "No".to_string()
                                }
                            }
                            "What is the TCP window size?" => {
                                return format!("{}", tcp.get_window());
                            }
                            _ => {}
                        }
                    }
                }
                IpNextHeaderProtocols::Udp => {
                    if let Some(udp) = UdpPacket::new(ipv4.payload()) {
                        match question {
                            "Is this packet a UDP datagram?" => {
                                return "Yes".to_string();
                            }
                            "What is the UDP source port?" => {
                                return format!("{}", udp.get_source());
                            }
                            "What is the UDP destination port?" => {
                                return format!("{}", udp.get_destination());
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
    }

    // Default answer if we couldn't determine
    "Unable to determine from packet data".to_string()
}

// Extract field value for a field finding task
fn extract_field_value(packet: &PacketData, field: &str) -> String {
    if let Some(ethernet) = EthernetPacket::new(&packet.data) {
        match field {
            "source_mac" => return ethernet.get_source().to_string(),
            "dest_mac" => return ethernet.get_destination().to_string(),
            "ethertype" => return format!("0x{:04x}", u16::from_be(ethernet.get_ethertype().0)),
            _ => {}
        }

        // Check for IPv4
        if let Some(ipv4) = Ipv4Packet::new(ethernet.payload()) {
            match field {
                "source_ip" => return ipv4.get_source().to_string(),
                "dest_ip" => return ipv4.get_destination().to_string(),
                "ttl" => return format!("{}", ipv4.get_ttl()),
                "protocol" => return format!("{}", ipv4.get_next_level_protocol().0),
                _ => {}
            }

            // Check TCP/UDP
            match ipv4.get_next_level_protocol() {
                IpNextHeaderProtocols::Tcp => {
                    if let Some(tcp) = TcpPacket::new(ipv4.payload()) {
                        match field {
                            "source_port" => return format!("{}", tcp.get_source()),
                            "dest_port" => return format!("{}", tcp.get_destination()),
                            "tcp_flags" => {
                                let mut flags = Vec::new();
                                if tcp.get_flags() & 0x02 != 0 { flags.push("SYN"); }
                                if tcp.get_flags() & 0x10 != 0 { flags.push("ACK"); }
                                if tcp.get_flags() & 0x01 != 0 { flags.push("FIN"); }
                                if tcp.get_flags() & 0x04 != 0 { flags.push("RST"); }
                                if tcp.get_flags() & 0x08 != 0 { flags.push("PSH"); }
                                if tcp.get_flags() & 0x20 != 0 { flags.push("URG"); }
                                return flags.join("|");
                            }
                            "tcp_window_size" => return format!("{}", tcp.get_window()),
                            _ => {}
                        }
                    }
                }
                IpNextHeaderProtocols::Udp => {
                    if let Some(udp) = UdpPacket::new(ipv4.payload()) {
                        match field {
                            "source_port" => return format!("{}", udp.get_source()),
                            "dest_port" => return format!("{}", udp.get_destination()),
                            "udp_length" => return format!("{}", udp.get_length()),
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
    }

    "unknown".to_string()
}

// Task creation functions
fn create_bmlm_task(
    packet: &PacketData,
    writer: &Arc<Mutex<BufWriter<File>>>,
    rng: &mut rand::rngs::ThreadRng,
) -> io::Result<()> {
    // Skip if packet is too large
    if packet.data.len() > 850 {
        return Ok(());
    }

    // Create a copy of packet data
    let packet_data = packet.data.to_vec();

    // Randomly mask 15% of bytes
    let mask_count = (packet_data.len() as f32 * 0.15) as usize;
    let mut masked_indices = Vec::new();
    let mut masked_values = Vec::new();

    // Pick random indices to mask
    let mut indices: Vec<usize> = (0..packet_data.len()).collect();
    indices.shuffle(rng);

    for &idx in indices.iter().take(mask_count) {
        masked_indices.push(idx);
        masked_values.push(packet_data[idx]);
    }

    // Sort indices for consistent output
    let mut sorted_pairs: Vec<(usize, u8)> = masked_indices.iter()
        .zip(masked_values.iter())
        .map(|(&idx, &val)| (idx, val))
        .collect();
    sorted_pairs.sort_by_key(|pair| pair.0);

    let sorted_indices: Vec<usize> = sorted_pairs.iter().map(|pair| pair.0).collect();
    let sorted_values: Vec<u8> = sorted_pairs.iter().map(|pair| pair.1).collect();

    // Create a task object
    let task = BMLMTask {
        task_type: TaskType::BMLM,
        pcap_bytes: packet_data,
        masked_indices: sorted_indices,
        masked_values: sorted_values,
    };

    // Thread-safe write to file
    let task_json = serde_json::to_string(&task)?;
    let mut writer_guard = writer.lock().unwrap();
    writeln!(writer_guard, "{}", task_json)?;

    Ok(())
}

fn create_next_byte_task(
    packet: &PacketData,
    writer: &Arc<Mutex<BufWriter<File>>>,
    rng: &mut rand::rngs::ThreadRng,
) -> io::Result<()> {
    // Skip if a packet is too small
    if packet.data.len() < 10 {
        return Ok(());
    }

    // Skip if packet is too large
    if packet.data.len() > 1000 {
        return Ok(());
    }

    // Choose a random position to predict (not the first byte)
    let target_pos = rng.random_range(1..packet.data.len());
    let input_data = packet.data[0..target_pos].to_vec();
    let target_byte = packet.data[target_pos];

    // Create a task object
    let task = NextByteTask {
        task_type: TaskType::NextBytePrediction,
        input_bytes: input_data,
        target_byte,
    };

    // Thread-safe write to file
    let task_json = serde_json::to_string(&task)?;
    let mut writer_guard = writer.lock().unwrap();
    writeln!(writer_guard, "{}", task_json)?;

    Ok(())
}

fn create_json_task(
    packet: &PacketData,
    writer: &Arc<Mutex<BufWriter<File>>>,
    rng: &mut rand::rngs::ThreadRng,
) -> io::Result<()> {
    // Skip if packet is too large
    if packet.data.len() > 500 {
        return Ok(());
    }

    // Save packet to a temporary file using a different approach
    let temp_file = format!("/tmp/packet_{}.pcap", rng.random::<u32>());

    // Create a temporary PCAP file manually
    {
        let mut temp_file_handle = File::create(&temp_file)?;

        // Write PCAP global header (simplified)
        let pcap_header = [
            0xd4, 0xc3, 0xb2, 0xa1, // magic number
            0x02, 0x00, 0x04, 0x00, // version
            0x00, 0x00, 0x00, 0x00, // timezone
            0x00, 0x00, 0x00, 0x00, // sigfigs
            0xff, 0xff, 0x00, 0x00, // snaplen
            0x01, 0x00, 0x00, 0x00  // linktype (Ethernet)
        ];
        temp_file_handle.write_all(&pcap_header)?;

        // Write packet header
        let timestamp_sec = 0u32.to_le_bytes();
        let timestamp_usec = 0u32.to_le_bytes();
        let packet_len = (packet.data.len() as u32).to_le_bytes();
        let packet_len_orig = packet_len.clone();

        temp_file_handle.write_all(&timestamp_sec)?;
        temp_file_handle.write_all(&timestamp_usec)?;
        temp_file_handle.write_all(&packet_len)?;
        temp_file_handle.write_all(&packet_len_orig)?;

        // Write packet data
        temp_file_handle.write_all(&packet.data)?;
    }

    // Run tshark to get JSON output
    let output = Command::new("tshark")
        .args(&["-r", &temp_file, "-T", "json", "-x"])
        .output()
        .expect("Failed to execute tshark");

    let json_str = String::from_utf8_lossy(&output.stdout).to_string();

    // Clean up temp file
    fs::remove_file(temp_file).ok();

    // Create a task object
    let task = JSONTask {
        task_type: TaskType::WiresharkJSON,
        pcap_bytes: packet.data.to_vec(),
        json_output: json_str,
    };

    // Thread-safe write to file
    let task_json = serde_json::to_string(&task)?;
    let mut writer_guard = writer.lock().unwrap();
    writeln!(writer_guard, "{}", task_json)?;

    Ok(())
}

fn create_qa_task(
    packet: &PacketData,
    writer: &Arc<Mutex<BufWriter<File>>>,
    rng: &mut rand::rngs::ThreadRng,
) -> io::Result<()> {
    // Skip if packet is too large
    if packet.data.len() > 800 {
        return Ok(());
    }

    // Pick a random question
    let question_idx = rng.random_range(0..QA_QUESTIONS.len());
    let question = QA_QUESTIONS[question_idx];

    // Generate answer based on packet analysis
    let answer = analyze_packet_for_qa(packet, question);

    // Create a task object
    let task = QATask {
        task_type: TaskType::QuestionAnswering,
        question: question.to_string(),
        pcap_bytes: packet.data.to_vec(),
        answer,
    };

    // Thread-safe write to file
    let task_json = serde_json::to_string(&task)?;
    let mut writer_guard = writer.lock().unwrap();
    writeln!(writer_guard, "{}", task_json)?;

    Ok(())
}

fn create_field_finding_task(
    packets: &[PacketData],
    writer: &Arc<Mutex<BufWriter<File>>>,
) -> io::Result<()> {
    // We need multiple packets for a flow
    if packets.len() < 3 {
        return Ok(());
    }

    // Combine packets into a flow
    let mut flow_data = Vec::new();
    for packet in packets {
        flow_data.extend_from_slice(&packet.data);
    }

    // Skip if flow is too large
    if flow_data.len() > 940 {
        return Ok(());
    }

    // Define fields we can extract
    let fields = [
        "source_mac", "dest_mac", "ethertype",
        "source_ip", "dest_ip", "ttl", "protocol",
        "source_port", "dest_port", "tcp_flags",
        "tcp_window_size", "udp_length"
    ];

    // Create thread-local RNG for field selection
    let mut rng = rand::rng();

    // For each packet in the flow, create a field finding task
    for (i, packet) in packets.iter().enumerate() {
        // Pick a random field
        let field_idx = rng.random_range(0..fields.len());
        let field = fields[field_idx];

        // Extract the field value
        let field_value = extract_field_value(packet, field);

        // Create a task object
        let task = FieldFindingTask {
            task_type: TaskType::FieldFinding,
            packet_id: i,
            field_name: field.to_string(),
            flow_bytes: flow_data.clone(),
            field_value,
        };

        // Thread-safe write to file
        let task_json = serde_json::to_string(&task)?;
        let mut writer_guard = writer.lock().unwrap();
        writeln!(writer_guard, "{}", task_json)?;
    }

    Ok(())
}

// Process a PCAP file using multithreading
fn process_pcap_file_mt(
    file_path: &Path,
    writers: Arc<ThreadSafeWriters>,
    progress: Arc<ProgressBar>,
) -> io::Result<()> {
    // Create a pcap capture from a file
    let mut capture = Capture::from_file(file_path).expect("Failed to open pcap file");

    // Collect packet data - solving the borrowing issue
    let mut packet_data = Vec::new();

    // Count total packets for progress reporting
    let mut packet_count = 0;

    // Use loop with match to avoid long-lived borrows
    loop {
        match capture.next_packet() {
            Ok(packet) => {
                packet_data.push(PacketData {
                    data: packet.data.to_vec(),
                });
                packet_count += 1;
            }
            Err(_) => break
        }
    }

    progress.set_length(packet_count as u64);
    progress.set_message(format!("Processing {} packets", packet_count));

    // Process packets in batches
    let batch_size = 10;
    let mut processed = 0;

    for chunk in packet_data.chunks(batch_size) {
        let chunk_vec = chunk.to_vec();

        // Process single packet tasks
        for packet in &chunk_vec {
            // Create thread-local RNG for this packet
            let mut rng = rand::rng();

            // Process tasks for this packet
            create_bmlm_task(packet, &writers.bmlm_writer, &mut rng)?;
            create_next_byte_task(packet, &writers.nbp_writer, &mut rng)?;
            create_json_task(packet, &writers.json_writer, &mut rng)?;
            create_qa_task(packet, &writers.qa_writer, &mut rng)?;

            processed += 1;
            if processed % 10 == 0 {
                progress.set_position(processed as u64);
            }
        }

        // Process flow-based task
        create_field_finding_task(&chunk_vec, &writers.field_writer)?;
    }

    progress.finish();
    Ok(())
}

// Load the list of processed files
fn load_processed_files(output_dir: &Path) -> io::Result<HashSet<String>> {
    let processed_files_path = output_dir.join("processed_files.txt");
    let mut processed_files = HashSet::new();

    // If the file exists, read it
    if processed_files_path.exists() {
        let file = File::open(&processed_files_path)?;
        let reader = BufReader::new(file);

        for line in reader.lines() {
            if let Ok(line) = line {
                if !line.trim().is_empty() {
                    processed_files.insert(line);
                }
            }
        }

        println!("Loaded {} previously processed files", processed_files.len());
    }

    Ok(processed_files)
}

// Save the list of processed files
fn save_processed_files(output_dir: &Path, processed_files: &HashSet<String>) -> io::Result<()> {
    let processed_files_path = output_dir.join("processed_files.txt");
    let mut file = File::create(&processed_files_path)?;

    for path in processed_files {
        writeln!(file, "{}", path)?;
    }

    Ok(())
}

// TaskGenerator implementation
impl TaskGenerator {
    fn new(output_dir: PathBuf) -> io::Result<Self> {
        fs::create_dir_all(&output_dir)?;

        // Initialize the processed files set from the record file
        let processed_files = Arc::new(Mutex::new(load_processed_files(&output_dir)?));

        // Create the thread-safe file writers
        let writers = ThreadSafeWriters {
            bmlm_writer: Arc::new(Mutex::new(BufWriter::new(File::create(output_dir.join("bmlm_tasks.jsonl"))?))),
            nbp_writer: Arc::new(Mutex::new(BufWriter::new(File::create(output_dir.join("nbp_tasks.jsonl"))?))),
            json_writer: Arc::new(Mutex::new(BufWriter::new(File::create(output_dir.join("json_tasks.jsonl"))?))),
            qa_writer: Arc::new(Mutex::new(BufWriter::new(File::create(output_dir.join("qa_tasks.jsonl"))?))),
            field_writer: Arc::new(Mutex::new(BufWriter::new(File::create(output_dir.join("field_tasks.jsonl"))?))),
        };

        Ok(Self {
            writers,
            processed_files,
            output_dir,
        })
    }

    // Process all PCAP files in a directory using multiple threads
    fn process_directory(&mut self, dir_path: &Path) -> io::Result<()> {
        // First, collect all the PCAP files that need to be processed
        let mut pcap_files = Vec::new();

        // Use a local reference to the processed_files HashSet
        let processed_files_lock = self.processed_files.lock().unwrap();

        for entry in fs::read_dir(dir_path)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() && path.extension().map_or(false, |ext| ext == "pcap") {
                let file_path_str = path.to_string_lossy().to_string();

                // Skip if the file has already been processed
                if !processed_files_lock.contains(&file_path_str) {
                    pcap_files.push(path);
                }
            }
        }

        // Display how many files will be skipped
        let skipped_count = processed_files_lock.len();
        if skipped_count > 0 {
            println!("Skipping {} already processed files", skipped_count);
        }

        // Release the lock
        drop(processed_files_lock);

        // Initialize multi-progress bar
        let multi_progress = Arc::new(MultiProgress::new());
        let main_progress = multi_progress.add(ProgressBar::new(pcap_files.len() as u64));
        main_progress.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
            .unwrap()
            .progress_chars("#>-"));
        main_progress.set_message("Processing PCAP files");

        // Clone necessary data for thread access
        let writers = Arc::new(self.writers.clone());
        let processed_files = Arc::clone(&self.processed_files);
        let output_dir = self.output_dir.clone();
        let main_progress_clone = main_progress.clone();

        // Process files in parallel
        pcap_files.into_par_iter()
            .with_max_len(1) // Process one file per thread
            .for_each(move |path| {
                let file_path_str = path.to_string_lossy().to_string();
                let file_name = path.file_name().unwrap_or_default().to_string_lossy().to_string();

                // Create a thread-specific progress indicator
                let thread_id = rayon::current_thread_index().unwrap_or(0);
                let thread_progress = Arc::new(ProgressBar::new(100));
                thread_progress.set_style(ProgressStyle::default_bar()
                    .template(&format!("Thread {}: [{{bar:30}}] {{pos}}% {{msg}}", thread_id))
                    .unwrap()
                    .progress_chars("=> "));
                thread_progress.set_message(format!("Processing {}", file_name));

                // Process the file
                match process_pcap_file_mt(&path, Arc::clone(&writers), thread_progress.clone()) {
                    Ok(_) => {
                        // Mark the file as processed
                        let mut processed_files_guard = processed_files.lock().unwrap();
                        processed_files_guard.insert(file_path_str.clone());

                        // Clone the HashSet for saving - this lets us drop the lock sooner
                        let files_to_save = processed_files_guard.clone();

                        // Drop the lock as soon as possible
                        drop(processed_files_guard);

                        // We don't want every thread to write to the file at the same time, 
                        // So we'll use a small sleep to stagger writes
                        std::thread::sleep(std::time::Duration::from_millis(10));

                        // Save the cloned set to file
                        let _ = save_processed_files(&output_dir, &files_to_save);
                    }
                    Err(e) => {
                        eprintln!("Error processing {}: {}", file_path_str, e);
                    }
                }

                // Update the main progress bar
                main_progress_clone.inc(1);
                thread_progress.finish_and_clear();
            });

        main_progress.finish_with_message("Processing complete");
        Ok(())
    }

    // Close all file writers
    fn close(&mut self) -> io::Result<()> {
        self.writers.bmlm_writer.lock().unwrap().flush()?;
        self.writers.nbp_writer.lock().unwrap().flush()?;
        self.writers.json_writer.lock().unwrap().flush()?;
        self.writers.qa_writer.lock().unwrap().flush()?;
        self.writers.field_writer.lock().unwrap().flush()?;
        Ok(())
    }
}

// Create a Python helper script for tokenization
fn create_python_helper(output_dir: &Path) -> io::Result<()> {
    let script = r#"#!/usr/bin/env python
import os
import json
import torch
from transformers import ByT5Tokenizer
from pathlib import Path
import sys

# Import custom tokenizers
sys.path.append('.')
try:
    from src.byteflow.tokenizer.pcap_tokenizer import PCAPTokenizer
    from tokenizer import HybridByT5PCAPTokenizer
    custom_tokenizer_available = True
except ImportError:
    print("Warning: Custom tokenizers not found, using standard ByT5 tokenizer")
    custom_tokenizer_available = False

def load_jsonl(file_path):
    """Load data from JSONL file"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def tokenize_bmlm_task(task, tokenizer):
    """Tokenize BMLM task for ByT5"""
    # Create masked version of pcap_bytes
    masked_bytes = task['pcap_bytes'].copy()
    for idx in task['masked_indices']:
        masked_bytes[idx] = 255  # Mask token

    # Format target as "idx:val,idx:val,..."
    target_text = ",".join([f"{idx}:{val}" for idx, val in zip(task['masked_indices'], task['masked_values'])])

    # Tokenize input and target
    input_tokens = tokenizer("<BMLM>" + bytes(masked_bytes).decode('latin1', errors='replace'), return_tensors="pt")
    target_tokens = tokenizer(target_text, return_tensors="pt")

    return {
        "input_ids": input_tokens.input_ids,
        "attention_mask": input_tokens.attention_mask,
        "labels": target_tokens.input_ids
    }

def tokenize_nbp_task(task, tokenizer):
    """Tokenize Next Byte Prediction task for ByT5"""
    # Tokenize input
    input_tokens = tokenizer("<NBP>" + bytes(task['input_bytes']).decode('latin1', errors='replace'), return_tensors="pt")

    # Target is a single byte
    target_text = str(task['target_byte'])
    target_tokens = tokenizer(target_text, return_tensors="pt")

    return {
        "input_ids": input_tokens.input_ids,
        "attention_mask": input_tokens.attention_mask,
        "labels": target_tokens.input_ids
    }

def tokenize_json_task(task, tokenizer):
    """Tokenize JSON task for ByT5"""
    # Tokenize input
    input_tokens = tokenizer("<JSON>" + bytes(task['pcap_bytes']).decode('latin1', errors='replace'), return_tensors="pt")

    # Target is the JSON string
    target_tokens = tokenizer(task['json_output'], return_tensors="pt")

    return {
        "input_ids": input_tokens.input_ids,
        "attention_mask": input_tokens.attention_mask,
        "labels": target_tokens.input_ids
    }

def tokenize_qa_task(task, tokenizer):
    """Tokenize Question Answering task for ByT5"""
    # Tokenize input (question + pcap bytes)
    input_text = f"<QA>{task['question']} " + bytes(task['pcap_bytes']).decode('latin1', errors='replace')
    input_tokens = tokenizer(input_text, return_tensors="pt")

    # Target is the answer
    target_tokens = tokenizer(task['answer'], return_tensors="pt")

    return {
        "input_ids": input_tokens.input_ids,
        "attention_mask": input_tokens.attention_mask,
        "labels": target_tokens.input_ids
    }

def tokenize_field_task(task, tokenizer):
    """Tokenize Field Finding task for ByT5"""
    # Tokenize input
    input_text = f"<FIELD_FINDING>{task['packet_id']} {task['field_name']} " + bytes(task['flow_bytes']).decode('latin1', errors='replace')
    input_tokens = tokenizer(input_text, return_tensors="pt")

    # Target is the field value
    target_tokens = tokenizer(task['field_value'], return_tensors="pt")

    return {
        "input_ids": input_tokens.input_ids,
        "attention_mask": input_tokens.attention_mask,
        "labels": target_tokens.input_ids
    }

def main():
    output_dir = Path("tokenized_data")
    output_dir.mkdir(exist_ok=True)

    # Initialize tokenizer
    if custom_tokenizer_available:
        print("Using custom HybridByT5PCAPTokenizer")
        tokenizer = HybridByT5PCAPTokenizer.from_pretrained("google/byt5-small")
    else:
        print("Using standard ByT5Tokenizer")
        tokenizer = ByT5Tokenizer.from_pretrained("google/byt5-small")

    # Process each task type
    task_processors = {
        "bmlm_tasks.jsonl": tokenize_bmlm_task,
        "nbp_tasks.jsonl": tokenize_nbp_task,
        "json_tasks.jsonl": tokenize_json_task,
        "qa_tasks.jsonl": tokenize_qa_task,
        "field_tasks.jsonl": tokenize_field_task,
    }

    for task_file, processor_func in task_processors.items():
        print(f"Processing {task_file}...")
        file_path = Path(task_file)
        if not file_path.exists():
            print(f"  File {task_file} not found, skipping")
            continue

        # Load data
        tasks = load_jsonl(file_path)
        print(f"  Found {len(tasks)} tasks")

        # Tokenize tasks
        tokenized_tasks = []
        for task in tasks:
            try:
                tokenized_task = processor_func(task, tokenizer)
                tokenized_tasks.append(tokenized_task)
            except Exception as e:
                print(f"  Error processing task: {e}")

        # Save as PyTorch tensor
        output_path = output_dir / f"{file_path.stem}.pt"
        torch.save(tokenized_tasks, output_path)
        print(f"  Saved {len(tokenized_tasks)} tokenized tasks to {output_path}")

if __name__ == "__main__":
    main()
"#;

    let script_path = output_dir.join("tokenize_for_byt5.py");
    let mut file = File::create(script_path)?;
    file.write_all(script.as_bytes())?;

    Ok(())
}

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 || args.len() > 4 {
        eprintln!("Usage: {} <pcap_directory> <output_directory> [num_threads]", args[0]);
        std::process::exit(1);
    }

    let pcap_dir = Path::new(&args[1]);
    let output_dir = Path::new(&args[2]);

    // Determine the number of threads to use (default to number of logical CPUs)
    let num_threads = if args.len() == 4 {
        args[3].parse::<usize>().unwrap_or_else(|_| {
            let num_cpus = num_cpus::get();
            println!("Invalid thread count, using system default: {}", num_cpus);
            num_cpus
        })
    } else {
        let num_cpus = num_cpus::get();
        println!("Using system default thread count: {}", num_cpus);
        num_cpus
    };

    // Configure thread pool
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();

    // Count the total number of PCAP files for progress reporting
    let total_pcap_files = fs::read_dir(pcap_dir)?
        .filter_map(Result::ok)
        .filter(|entry| {
            entry.path().is_file() &&
                entry.path().extension().map_or(false, |ext| ext == "pcap")
        })
        .count();

    println!("Found {} PCAP files in directory", total_pcap_files);
    println!("Processing with {} threads", num_threads);

    let mut generator = TaskGenerator::new(output_dir.to_path_buf())?;
    generator.process_directory(pcap_dir)?;
    generator.close()?;

    // Create a Python helper script for tokenization
    create_python_helper(output_dir)?;

    println!("Processing complete. Task data saved to {:?}", output_dir);
    println!("Use tokenize_for_byt5.py script to convert JSON to PyTorch tensors");
    Ok(())
}