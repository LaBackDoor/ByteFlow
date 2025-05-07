import logging
import os
import struct
from collections import defaultdict

from scapy.all import rdpcap
from scapy.layers.l2 import Ether


class PCAPTokenizer:
    def __init__(self, vocab_size=277, offset=3):
        self.vocab_size = vocab_size
        self.offset = offset
        self.special_tokens = {
            'packet_start': 0x100 + offset,
            'end': 0x101 + offset,
        }
        if vocab_size < 258:
            raise ValueError(f"Vocab size {vocab_size} is too small. Minimum is 258.")
        self.hex_to_token = {i: i + offset for i in range(256)}
        self.allocated_tokens = set(range(offset, 256 + offset))
        self.allocated_tokens.update(self.special_tokens.values())
        self.link_types = {
            0: self._allocate_token(), 1: self._allocate_token(),
            8: self._allocate_token(), 9: self._allocate_token(),
            10: self._allocate_token(), 101: self._allocate_token(),
            105: self._allocate_token(), 113: self._allocate_token(),
            127: self._allocate_token(),
        }
        self.flows = defaultdict(list)
        self.logger = logging.getLogger('PCAPTokenizer')

    def _allocate_token(self):
        for token_id in range(256 + self.offset, self.vocab_size + self.offset):
            if token_id not in self.allocated_tokens:
                self.allocated_tokens.add(token_id)
                return token_id
        raise ValueError(f"Token vocabulary limit of {self.vocab_size} exceeded")

    def tokenize_pcap(self, pcap_file):
        """
        Tokenize a PCAP file.
        Args:
            pcap_file: Path to the PCAP file.
        Returns:
            Dictionary mapping flow identifiers to token lists
        """
        try:
            packets_from_file = rdpcap(pcap_file)
        except Exception as e:
            self.logger.error(f"Error reading PCAP file '{pcap_file}': {e}")
            return {}

        if not packets_from_file:
            self.logger.warning(f"No packets found in PCAP file '{pcap_file}'.")
            return {}

        self.flows = defaultdict(list)  # Reset flows for this tokenization run

        base_name = os.path.basename(pcap_file)
        flow_id = f"{base_name}"

        sorted_packets = sorted(packets_from_file, key=lambda p: float(p.time))
        self.flows[flow_id] = sorted_packets

        # Tokenize each flow
        tokenized_flows_output = {}
        for flow_id, flow_packets_list in self.flows.items():
            if not flow_packets_list:
                self.logger.warning(f"Flow {flow_id} has no packets after extraction. Skipping.")
                continue
            tokenized_flows_output[flow_id] = self._tokenize_flow(flow_packets_list)

        return tokenized_flows_output


    def _tokenize_flow(self, packets_in_flow):
        tokens = []
        prev_time = None
        for packet in packets_in_flow:
            tokens.append(self.special_tokens['packet_start'])
            link_type_token = self._get_link_type_token(packet)
            tokens.append(link_type_token)
            curr_time = float(packet.time)
            time_interval = curr_time - prev_time if prev_time is not None else 0.0
            if time_interval < 0:
                self.logger.warning(f"Negative time interval ({time_interval}s) detected. "
                                    "Ensure packets in flows are chronologically sorted.")
                # Optionally, clamp to 0 or handle as an error
                time_interval = 0.0
            time_tokens = self._encode_time_interval(time_interval)
            tokens.extend(time_tokens)
            prev_time = curr_time
            raw_data = bytes(packet)
            hex_tokens = self._encode_packet_data(raw_data)
            tokens.extend(hex_tokens)
        tokens.append(self.special_tokens['end'])
        return tokens

    def _get_link_type_token(self, packet):
        link_type = None
        if Ether in packet:
            link_type = 1
        elif hasattr(packet, 'linktype'):
            link_type = packet.linktype
        if link_type in self.link_types: return self.link_types[link_type]
        if link_type is not None:
            try:
                self.link_types[link_type] = self._allocate_token()
                return self.link_types[link_type]
            except ValueError:
                self.logger.warning(f"Vocab limit for link type {link_type}, using default.")
        return self.link_types.get(1, self._allocate_token())  # Default to Ethernet

    def _encode_time_interval(self, time_interval):
        packed = struct.pack('!d', time_interval)
        return [self.hex_to_token[byte] for byte in packed]

    def _encode_packet_data(self, raw_data):
        return [self.hex_to_token[byte] for byte in raw_data]

    # decode_flow and flows_to_pcap remain the same as you provided
    def decode_flow(self, tokens):
        token_to_link_type = {v: k for k, v in self.link_types.items()}
        packets_info = []  # Renamed
        i = 0
        while i < len(tokens):
            if tokens[i] == self.special_tokens['packet_start']:
                i += 1
                if i >= len(tokens): self.logger.warning("Unexpected end after packet start"); break
                link_type_token = tokens[i]
                link_type = token_to_link_type.get(link_type_token, 1)
                i += 1
                if i + 8 > len(tokens): self.logger.warning("Not enough tokens for time"); break
                time_tokens = tokens[i:i + 8]
                time_bytes = bytes(
                    [(t - self.offset) if self.offset <= t < (256 + self.offset) else 0 for t in time_tokens])
                time_interval = struct.unpack('!d', time_bytes)[0]
                i += 8
                packet_data_bytes = []  # Renamed
                while (i < len(tokens) and
                       tokens[i] != self.special_tokens['packet_start'] and
                       tokens[i] != self.special_tokens['end']):
                    byte_value = (tokens[i] - self.offset) if self.offset <= tokens[i] < (256 + self.offset) else 0
                    packet_data_bytes.append(byte_value)
                    i += 1
                packets_info.append((link_type, time_interval, bytes(packet_data_bytes)))
            elif tokens[i] == self.special_tokens['end']:
                break
            else:
                self.logger.warning(f"Unexpected token {tokens[i]} at pos {i}")
                i += 1
        return packets_info

    def flows_to_pcap(self, tokenized_flows_dict, output_file_path):  # Renamed
        from scapy.utils import wrpcap
        reconstructed_packets = []  # Renamed
        for flow_id, tokens_list in tokenized_flows_dict.items():  # Renamed
            decoded_packets_info = self.decode_flow(tokens_list)
            for link_type, time_interval, packet_byte_data in decoded_packets_info:  # Renamed
                try:
                    packet_obj = Ether(packet_byte_data)
                    reconstructed_packets.append(packet_obj)
                except Exception as e:
                    self.logger.warning(f"Failed to reconstruct packet from flow {flow_id}: {e}")
        if reconstructed_packets:
            wrpcap(output_file_path, reconstructed_packets)
        return len(reconstructed_packets)