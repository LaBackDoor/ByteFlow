import os
import tempfile
from transformers import ByT5Tokenizer

from src.byteflow.tokenizer.pcap_tokenizer import PCAPTokenizer


class HybridByT5PCAPTokenizer(ByT5Tokenizer):
    """
    A tokenizer that combines ByT5's byte-level tokenization with PCAP tokenization,
    supporting mixed inputs of text and PCAP data.
    """

    def __init__(
            self,
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            extra_ids=0,
            additional_special_tokens=None,
            pcap_vocab_size=280,  # Increased for new special tokens
            **kwargs,
    ):
        # Initialize with special tokens for content boundaries
        special_tokens = [
            "<pcap_start>",
            "<pcap_end>",
            "<text_start>",
            "<text_end>",
            "<pcap_attachment>",
            "<packet_start>",
            "<packet_end>",
            "<flow_end>",      # New token for flow end
            "<field_sep>"      # Field separator token
        ]

        if additional_special_tokens:
            additional_special_tokens.extend(special_tokens)
        else:
            additional_special_tokens = special_tokens

        # Initialize the ByT5Tokenizer first
        super().__init__(
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        # Initialize the PCAP tokenizer
        self.pcap_tokenizer = PCAPTokenizer(vocab_size=pcap_vocab_size)

        # Store the token IDs for easy access
        self.pcap_start_token_id = self.convert_tokens_to_ids("<pcap_start>")
        self.pcap_end_token_id = self.convert_tokens_to_ids("<pcap_end>")
        self.text_start_token_id = self.convert_tokens_to_ids("<text_start>")
        self.text_end_token_id = self.convert_tokens_to_ids("<text_end>")
        self.pcap_attachment_token_id = self.convert_tokens_to_ids("<pcap_attachment>")
        self.field_sep_token_id = self.convert_tokens_to_ids("<field_sep>")
        self.packet_start_token_id = self.convert_tokens_to_ids("<packet_start>")
        self.packet_end_token_id = self.convert_tokens_to_ids("<packet_end>")
        self.flow_end_token_id = self.convert_tokens_to_ids("<flow_end>")

        # Map between PCAPTokenizer special tokens and our special token IDs
        self.pcap_to_byt5_token_map = {
            self.pcap_tokenizer.special_tokens['packet_start']: self.packet_start_token_id,
            self.pcap_tokenizer.special_tokens['packet_end']: self.packet_end_token_id,
            self.pcap_tokenizer.special_tokens['flow_end']: self.flow_end_token_id,
            self.pcap_tokenizer.special_tokens['field_sep']: self.field_sep_token_id
        }

        # Add tokens for all link types
        for link_type, token_id in self.pcap_tokenizer.link_types.items():
            token_str = f"<link_type_{link_type}>"
            self.add_tokens([token_str])
            self.pcap_to_byt5_token_map[token_id] = self.convert_tokens_to_ids(token_str)

    def encode_mixed_input(self, text=None, pcap_bytes=None, pcap_file_path=None):
        """
        Encode mixed input of text and PCAP data.

        Args:
            text: Text content (optional)
            pcap_bytes: Raw PCAP bytes (optional)
            pcap_file_path: Path to a PCAP file (optional)

        Returns:
            Token IDs for the mixed input
        """
        token_ids = []

        # Handle text if provided
        if text is not None:
            token_ids.append(self.text_start_token_id)
            text_tokens = super().encode(text, add_special_tokens=False)
            token_ids.extend(text_tokens)
            token_ids.append(self.text_end_token_id)

        # Handle PCAP bytes if provided
        if pcap_bytes is not None:
            token_ids.append(self.pcap_start_token_id)
            pcap_tokens = self._tokenize_pcap_bytes(pcap_bytes)
            token_ids.extend(pcap_tokens)
            token_ids.append(self.pcap_end_token_id)

        # Handle PCAP file if provided
        if pcap_file_path:
            token_ids.append(self.pcap_attachment_token_id)
            pcap_tokens = self._tokenize_pcap_file(pcap_file_path)
            token_ids.extend(pcap_tokens)
            token_ids.append(self.pcap_end_token_id)

        return token_ids

    def _tokenize_pcap_bytes(self, pcap_bytes):
        """
        Tokenize raw PCAP bytes.

        Args:
            pcap_bytes: Raw PCAP data as bytes

        Returns:
            Tokenized representation of the PCAP data
        """
        # Write bytes to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.pcap', delete=False) as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(pcap_bytes)

        # Tokenize the temporary file
        try:
            tokens = self._tokenize_pcap_file(temp_file_path)
        finally:
            os.unlink(temp_file_path)

        return tokens

    def _tokenize_pcap_file(self, pcap_file_path):
        """
        Tokenize a PCAP file.

        Args:
            pcap_file_path: Path to the PCAP file

        Returns:
            Tokenized representation of the PCAP data
        """
        try:
            # Get tokenized flows from the PCAP tokenizer
            tokenized_flows = self.pcap_tokenizer.tokenize_pcap(pcap_file_path)

            # Convert the PCAP tokens to our token space
            converted_tokens = []

            for flow_id, tokens in tokenized_flows.items():
                flow_tokens = []

                for token in tokens:
                    if token in self.pcap_to_byt5_token_map:
                        # Special token mapping
                        flow_tokens.append(self.pcap_to_byt5_token_map[token])
                    elif token < 256:
                        # Byte tokens (0-255) are the same in both tokenizers
                        flow_tokens.append(token + self.offset)  # Apply ByT5 offset
                    else:
                        # Fallback for unknown tokens
                        flow_tokens.append(self.unk_token_id)

                converted_tokens.extend(flow_tokens)

            return converted_tokens
        except Exception as e:
            # Log error but don't crash
            import logging
            logging.error(f"Error tokenizing PCAP file {pcap_file_path}: {e}")
            # Return empty token list instead of crashing
            return []

    def decode_mixed_input(self, token_ids):
        """
        Decode token IDs back to text and PCAP data.

        Args:
            token_ids: List of token IDs

        Returns:
            Dict with 'text' and/or 'pcap_data' keys
        """
        result = {}
        i = 0

        while i < len(token_ids):
            if token_ids[i] == self.text_start_token_id:
                # Extract text content
                text_tokens = []
                i += 1  # Skip the start token

                while i < len(token_ids) and token_ids[i] != self.text_end_token_id:
                    text_tokens.append(token_ids[i])
                    i += 1

                if i < len(token_ids):
                    i += 1  # Skip the end token

                # Decode text
                text = super().decode(text_tokens)
                result['text'] = text

            elif token_ids[i] == self.pcap_start_token_id or token_ids[i] == self.pcap_attachment_token_id:
                # Extract PCAP content
                pcap_tokens = []
                i += 1  # Skip the start token

                while i < len(token_ids) and token_ids[i] != self.pcap_end_token_id:
                    pcap_tokens.append(token_ids[i])
                    i += 1

                if i < len(token_ids):
                    i += 1  # Skip the end token

                # Convert our tokens back to PCAP tokenizer tokens
                rev_map = {v: k for k, v in self.pcap_to_byt5_token_map.items()}
                pcap_native_tokens = []

                for token in pcap_tokens:
                    if token in rev_map:
                        # Special token mapping
                        pcap_native_tokens.append(rev_map[token])
                    elif self.offset <= token < self.offset + 256:
                        # Byte tokens
                        pcap_native_tokens.append(token - self.offset)
                    else:
                        # Unknown tokens
                        pcap_native_tokens.append(0)  # Default to null byte

                # Use PCAPTokenizer to decode the tokens
                pcap_data = self.pcap_tokenizer.decode_flow(pcap_native_tokens)

                if token_ids[i - 1] == self.pcap_end_token_id:  # Check if we found the end token
                    result['pcap_data'] = pcap_data
            else:
                # Skip unknown tokens
                i += 1

        return result

    # Add methods to handle the different input scenarios
    def tokenize_text_with_pcap(self, text, pcap_file_path):
        """Text with a PCAP file attachment"""
        if not os.path.exists(pcap_file_path):
            raise FileNotFoundError(f"PCAP file not found: {pcap_file_path}")

        token_ids = []

        # Only add text tokens if there is actual text content
        if text:
            token_ids.append(self.text_start_token_id)
            text_tokens = super().encode(text, add_special_tokens=False)
            token_ids.extend(text_tokens)
            token_ids.append(self.text_end_token_id)

        # Add PCAP tokens
        token_ids.append(self.pcap_attachment_token_id)
        pcap_tokens = self._tokenize_pcap_file(pcap_file_path)
        token_ids.extend(pcap_tokens)
        token_ids.append(self.pcap_end_token_id)

        return token_ids

    def tokenize_text_followed_by_pcap(self, text, pcap_bytes):
        """Text followed by PCAP bytes"""
        return self.encode_mixed_input(text=text, pcap_bytes=pcap_bytes)

    def tokenize_pcap_followed_by_text(self, pcap_bytes, text):
        """PCAP bytes followed by text"""
        # Note: The order is preserved in the token sequence
        return self.encode_mixed_input(pcap_bytes=pcap_bytes, text=text)