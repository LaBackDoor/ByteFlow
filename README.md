
# ByteFlow

ByteFlow is a deep learning framework designed for processing and understanding network traffic (PCAP files), potentially in combination with textual data. It leverages a novel 2D-aware Transformer architecture to effectively model the structured nature of network packets.

## Key Features

* **Advanced PCAP Processing**: Includes a Rust-based utility (`pcap_flow_splitter`) to efficiently split large PCAP files into individual network flows and perform deduplication. [cite: 1]
* **Hybrid Tokenization**: Features a `HybridByT5PCAPTokenizer` that converts raw PCAP data (packet headers and fields) and text into a unified token sequence. [cite: 2, 3] This tokenizer generates 2D positional indices to preserve the inherent tabular structure of packet data. [cite: 2]
* **2D-Aware Transformer Model**: The core of ByteFlow is a ByT5-like model implemented in PyTorch. It incorporates:
    * 2D relative position biases to understand spatial relationships in the tokenized PCAP data. [cite: 5, 10]
    * A routing network within its attention mechanism, allowing the model to dynamically process information from different perspectives (e.g., row-wise and permuted column-wise views of the data). [cite: 5, 10]
* **Mixed Data Training**: Supports pre-training on interleaved datasets, combining large text corpora (like C4) with network traffic data. [cite: 6]

## Goal

ByteFlow aims to enable deep learning models to achieve a nuanced understanding of network traffic by explicitly representing and processing its 2D structure, leading to better performance on downstream tasks like traffic analysis, intrusion detection, and network monitoring.