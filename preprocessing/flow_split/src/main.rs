use clap::{Arg, Command};
use indicatif::{ProgressBar, ProgressStyle};
use pcap::{Capture, Offline, Packet as PcapPacket};
use pnet_packet::{ethernet::{EthernetPacket, EtherTypes}, ip::IpNextHeaderProtocols, ipv4::Ipv4Packet, ipv6::Ipv6Packet, tcp::TcpPacket, udp::UdpPacket, Packet};
use rayon::prelude::*;
use std::{
    collections::HashSet,
    fs,
    io::{self},
    net::IpAddr,
    path::{Path, PathBuf, MAIN_SEPARATOR},
};
use walkdir::WalkDir;

// A canonical, bidirectional 5‑tuple key:
type FlowKey = (IpAddr, u16, IpAddr, u16, u8);

fn main() -> io::Result<()> {
    let matches = Command::new("pcap_flow_splitter")
        .version("0.1.0")
        .author("You <you@example.com>")
        .about("Splits each TCP/UDP flow in PCAPs into its own file, multi‑threaded")
        .arg(
            Arg::new("input_dir")
                .short('i')
                .long("input")
                .value_name("INPUT_DIR")
                .help("Directory tree to scan for .pcap/.pcapng")
                .required(true),
        )
        .arg(
            Arg::new("output_dir")
                .short('o')
                .long("output")
                .value_name("OUTPUT_DIR")
                .help("Where to write per‑flow pcaps")
                .required(true),
        )
        .get_matches();

    let input_dir = Path::new(matches.get_one::<String>("input_dir").unwrap());
    let output_dir = Path::new(matches.get_one::<String>("output_dir").unwrap());
    fs::create_dir_all(output_dir)?;

    // 1) Gather all .pcap/.pcapng paths
    let files: Vec<PathBuf> = WalkDir::new(input_dir)
        .into_iter()
        .filter_map(Result::ok)
        .map(|e| e.path().to_path_buf())
        .filter(|p| {
            p.is_file()
                && p.extension()
                    .and_then(|e| e.to_str())
                    .map(|s| {
                        let s = s.to_lowercase();
                        s == "pcap" || s == "pcapng"
                    })
                    .unwrap_or(false)
        })
        .collect();

    // 2) Progress bar
    let pb = ProgressBar::new(files.len() as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} files",
        )
        .unwrap()
        .progress_chars("#>-"),
    );

    // 3) Parallel processing
    files.par_iter().for_each(|path| {
        if let Err(err) = process_file(path, input_dir, output_dir) {
            eprintln!("Error on {:?}: {}", path, err);
        }
        pb.inc(1);
    });

    pb.finish_with_message("done");
    Ok(())
}

/// Handle one PCAP file: find its flows and dump each flow
fn process_file(path: &PathBuf, input_dir: &Path, output_dir: &Path) -> io::Result<()> {
    // Build a sanitized base from the PCAP’s relative path:
    let rel = path.strip_prefix(input_dir).unwrap();
    let base = rel
        .with_extension("")                                  // drop extension
        .to_string_lossy()                                   // e.g. "subdir/capture"
        .chars()
        .map(|c| if c == MAIN_SEPARATOR { '_' } else { c })  // replace separators
        .collect::<String>();

    // 1) Discover unique flows
    let flows = extract_flows(path)?;

    // 2) Dump each flow
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
    Ok(())
}

/// First pass: collect unique canonical FlowKeys
fn extract_flows(path: &Path) -> io::Result<HashSet<FlowKey>> {
    let mut cap = Capture::<Offline>::from_file(path)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    let mut set = HashSet::new();
    while let Ok(pkt) = cap.next_packet() {
        if let Some(key) = get_flow_key(&pkt) {
            set.insert(key);
        }
    }
    Ok(set)
}

/// Second pass: dump matching packets into `out_path`
fn write_flow(src: &Path, flow: &FlowKey, out_path: &Path) -> io::Result<()> {
    let mut cap = Capture::<Offline>::from_file(src)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    let mut dumper = cap
        .savefile(out_path)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    while let Ok(pkt) = cap.next_packet() {
        if let Some(key) = get_flow_key(&pkt) {
            if &key == flow {
                dumper.write(&pkt);
            }
        }
    }
    Ok(())
}

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

/// Order endpoints so A↔B == B↔A
fn canonicalize(a_ip: IpAddr, a_port: u16, b_ip: IpAddr, b_port: u16, proto: u8) -> FlowKey {
    if (a_ip, a_port) <= (b_ip, b_port) {
        (a_ip, a_port, b_ip, b_port, proto)
    } else {
        (b_ip, b_port, a_ip, a_port, proto)
    }
}

fn proto_name(n: u8) -> &'static str {
    match n {
        6  => "TCP",
        17 => "UDP",
        _  => "UNK",
    }
}

/// Make an IP string safe for filenames (e.g. IPv6 “:” → “_”)
fn sanitize_ip(ip: &IpAddr) -> String {
    ip.to_string()
        .chars()
        .map(|c| if c == ':' || c == '/' { '_' } else { c })
        .collect()
}
