use clap::{Arg, Command};
use indicatif::{ProgressBar, ProgressStyle};
use pcap::{Capture, Offline, Packet as PcapPacket};
use pnet_packet::{ethernet::{EthernetPacket, EtherTypes}, ip::IpNextHeaderProtocols, ipv4::Ipv4Packet, ipv6::Ipv6Packet, tcp::TcpPacket, udp::UdpPacket, Packet};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    net::IpAddr,
    path::{Path, PathBuf, MAIN_SEPARATOR},
    sync::{Arc, Mutex},
};
use walkdir::WalkDir;

// A canonical, bidirectional 5‑tuple key:
type FlowKey = (IpAddr, u16, IpAddr, u16, u8);

fn main() -> io::Result<()> {
    let matches = Command::new("pcap_flow_splitter")
        .version("0.1.0")
        .author("You <you@example.com>")
        .about("Splits each TCP/UDP flow in PCAPs into its own file, supports resumable processing")
        .arg(
            Arg::new("input_dir").short('i').long("input").value_name("INPUT_DIR").help("Directory tree to scan for .pcap/.pcapng").required(true),
        )
        .arg(
            Arg::new("output_dir").short('o').long("output").value_name("OUTPUT_DIR").help("Where to write per‑flow pcaps and resume log").required(true),
        )
        .arg(
            Arg::new("project_dir")
                .short('p')
                .long("project")
                .value_name("PROJECT_DIR")
                .help("Directory to write logs and reports")
                .required(true),
        )
        .get_matches();

    let input_dir = Path::new(matches.get_one::<String>("input_dir").unwrap());
    let output_dir = Path::new(matches.get_one::<String>("output_dir").unwrap());
    let project_dir = Path::new(matches.get_one::<String>("project_dir").unwrap());
    fs::create_dir_all(output_dir)?;
    fs::create_dir_all(project_dir)?;

    // Path to resume log:
    let log_path = project_dir.join("processed.txt");
    // Read an existing processed set
    let mut processed_set = HashSet::new();
    if log_path.exists() {
        let contents = fs::read_to_string(&log_path)?;
        for line in contents.lines() {
            processed_set.insert(line.to_string());
        }
    }
    // Open resume log for append
    let log_writer = Arc::new(Mutex::new(
        OpenOptions::new().create(true).append(true).open(&log_path)?
    ));

    // Gather all files to process (skip already processed)
    let mut files_to_process = Vec::new();
    for entry in WalkDir::new(input_dir).into_iter().filter_map(Result::ok) {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
        if ext != "pcap" && ext != "pcapng" {
            continue;
        }
        let rel = path.strip_prefix(input_dir).unwrap().to_string_lossy().to_string();
        if processed_set.contains(&rel) {
            continue;
        }
        files_to_process.push(path.to_path_buf());
        
    }

    // Progress bar
     let pb = ProgressBar::new(files_to_process.len() as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} files",
        )
        .unwrap()
        .progress_chars("#>-"),
    );

    // Parallel processing
    files_to_process.par_iter().for_each(|path| {
        let rel = path.strip_prefix(input_dir).unwrap().to_string_lossy().to_string();
        if let Err(err) = process_file(path, input_dir, output_dir) {
            eprintln!("Error processing {}: {}", rel, err);
        } else {
            // Append to resume log
            if let Ok(mut w) = log_writer.lock() {
                let _ = writeln!(w, "{}", rel);
                let _ = w.flush();
            }
        }
        pb.inc(1);
    });

    pb.finish_with_message("Splitting complete");
    // Deduplication and reporting
    dedupe_and_report(output_dir, project_dir)
}

/// Handle one PCAP file: extract flows and dump into separate PCAPs
fn process_file(path: &Path, input_dir: &Path, output_dir: &Path) -> io::Result<()> {
    // Sanitize base name from a relative path
    let rel = path.strip_prefix(input_dir).unwrap();
    let base = rel
        .with_extension("")
        .to_string_lossy()
        .chars()
        .map(|c| if c == MAIN_SEPARATOR { '_' } else { c })
        .collect::<String>();

    // 1) Extract unique flows
    let flows = extract_flows(path)?;

    // 2) Write each flow
    for flow in &flows {
        let flow_name = format!(
            "{proto}_{src}_{sport}_{dst}_{dport}.pcap",
            proto = proto_name(flow.4),
            src   = sanitize_ip(&flow.0),
            sport = flow.1,
            dst   = sanitize_ip(&flow.2),
            dport = flow.3,
        );
        let out_fname = format!("{}__{}", base, flow_name);
        let out_path = output_dir.join(out_fname);
        write_flow(path, flow, &out_path)?;
    }

    // Drop flows to free memory immediately
    drop(flows);
    Ok(())
}

/// First pass: collect unique flows from a PCAP
fn extract_flows(path: &Path) -> io::Result<HashSet<FlowKey>> {
    let mut cap = Capture::<Offline>::from_file(path)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    let mut set = HashSet::new();
    while let Ok(pkt) = cap.next_packet() {
        if let Some(k) = get_flow_key(&pkt) {
            set.insert(k);
        }
    }
    Ok(set)
}

/// Second pass: dump packets matching a flow into its own file
fn write_flow(src: &Path, flow: &FlowKey, out_path: &Path) -> io::Result<()> {
    let mut cap = Capture::<Offline>::from_file(src)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    let mut dumper = cap.savefile(out_path)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    while let Ok(pkt) = cap.next_packet() {
        if let Some(k) = get_flow_key(&pkt) {
            if &k == flow {
                dumper.write(&pkt);
            }
        }
    }
    Ok(())
}

/// Deduplicate files, log deletions, and report sizes
fn dedupe_and_report(output_dir: &Path, project_dir: &Path) -> io::Result<()> {
    let mut seen: HashMap<String, PathBuf> = HashMap::new();
    let mut del_log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(project_dir.join("deleted_flows.txt"))?;
    let mut size_csv = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(project_dir.join("flow_sizes.csv"))?;
    writeln!(size_csv, "filename,size_bytes")?;

    let mut largest: Option<(PathBuf, u64)> = None;
    for entry in WalkDir::new(output_dir).into_iter().filter_map(Result::ok) {
        let p = entry.path();
        if p.is_file() && p.extension().and_then(|e| e.to_str()).map(|s| s.eq_ignore_ascii_case("pcap")).unwrap_or(false) {
            let rel = p.strip_prefix(output_dir).unwrap().to_string_lossy().to_string();
            let sz = fs::metadata(p)?.len();
            writeln!(size_csv, "{},{}", rel, sz)?;
            if largest.as_ref().map_or(true, |(_, m)| sz > *m) {
                largest = Some((PathBuf::from(&rel), sz));
            }
            // compute SHA256
            let mut f = File::open(p)?;
            let mut hasher = Sha256::new();
            let mut buf = [0u8; 8192];
            loop {
                let n = f.read(&mut buf)?;
                if n == 0 { break; }
                hasher.update(&buf[..n]);
            }
            let hash_bytes = hasher.finalize();
            let hash_str = hash_bytes.iter().map(|b| format!("{:02x}", b)).collect::<String>();
            if let Some(_orig) = seen.insert(hash_str, PathBuf::from(&rel)) {
                fs::remove_file(p)?;
                writeln!(del_log, "{}", rel)?;
            }
        }
    }
    if let Some((path, sz)) = largest {
        let mut out = File::create(project_dir.join("largest_flow.txt"))?;
        writeln!(out, "{},{}", path.display(), sz)?;
    }
    Ok(())
}

/// Parse a packet to its canonical 5-tuple key
/// Parse a packet into a 5‑tuple, all in one scope.
fn get_flow_key(pkt: &PcapPacket) -> Option<FlowKey> {
    let eth = EthernetPacket::new(pkt.data)?;
    match eth.get_ethertype() {
        EtherTypes::Ipv4 => {
            let ip4 = Ipv4Packet::new(eth.payload())?;
            let src = IpAddr::V4(ip4.get_source());
            let dst = IpAddr::V4(ip4.get_destination());
            match ip4.get_next_level_protocol() {
                IpNextHeaderProtocols::Tcp => {
                    let tcp = TcpPacket::new(ip4.payload())?;
                    Some(canonicalize(
                        src,
                        tcp.get_source(),
                        dst,
                        tcp.get_destination(),
                        IpNextHeaderProtocols::Tcp.0,
                    ))
                }
                IpNextHeaderProtocols::Udp => {
                    let udp = UdpPacket::new(ip4.payload())?;
                    Some(canonicalize(
                        src,
                        udp.get_source(),
                        dst,
                        udp.get_destination(),
                        IpNextHeaderProtocols::Udp.0,
                    ))
                }
                _ => None,
            }
        }
        EtherTypes::Ipv6 => {
            let ip6 = Ipv6Packet::new(eth.payload())?;
            let src = IpAddr::V6(ip6.get_source());
            let dst = IpAddr::V6(ip6.get_destination());
            match ip6.get_next_header() {
                IpNextHeaderProtocols::Tcp => {
                    let tcp = TcpPacket::new(ip6.payload())?;
                    Some(canonicalize(
                        src,
                        tcp.get_source(),
                        dst,
                        tcp.get_destination(),
                        IpNextHeaderProtocols::Tcp.0,
                    ))
                }
                IpNextHeaderProtocols::Udp => {
                    let udp = UdpPacket::new(ip6.payload())?;
                    Some(canonicalize(
                        src,
                        udp.get_source(),
                        dst,
                        udp.get_destination(),
                        IpNextHeaderProtocols::Udp.0,
                    ))
                }
                _ => None,
            }
        }
        _ => None,
    }
}


fn canonicalize(a_ip: IpAddr, a_port: u16, b_ip: IpAddr, b_port: u16, proto: u8) -> FlowKey {
    if (a_ip, a_port) <= (b_ip, b_port) {
        (a_ip, a_port, b_ip, b_port, proto)
    } else {
        (b_ip, b_port, a_ip, a_port, proto)
    }
}

fn proto_name(n: u8) -> &'static str {
    match n {
        6 => "TCP",
        17 => "UDP",
        _ => "UNK",
    }
}

fn sanitize_ip(ip: &IpAddr) -> String {
    ip.to_string().chars().map(|c| if c == ':' || c == '/' { '_' } else { c }).collect()
}
