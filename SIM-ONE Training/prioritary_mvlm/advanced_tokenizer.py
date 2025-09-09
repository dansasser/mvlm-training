"""
Advanced BPE Tokenizer with Biblical Vocabulary for SIM-ONE Transformer
Implements efficient subword tokenization with specialized biblical terms
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
import pickle


class BiblicalBPETokenizer:
    """
    Advanced BPE tokenizer optimized for biblical and theological text.
    Features:
    - Byte Pair Encoding for efficient subword tokenization
    - Specialized biblical vocabulary 
    - Subword regularization
    - Unicode normalization
    - Fast encoding/decoding
    """
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []
        self.bpe_ranks = {}
        
        # Special tokens
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1, 
            '<bos>': 2,
            '<eos>': 3,
            '<mask>': 4,
            '<sep>': 5,
            # Biblical special tokens
            '<verse>': 6,
            '<chapter>': 7,
            '<book>': 8,
            '<quote>': 9,
            '<prayer>': 10,
            '<prophecy>': 11,
            '<parable>': 12,
            '<genealogy>': 13,
            '<law>': 14,
            '<psalm>': 15
        }
        
        self.pad_token_id = self.special_tokens['<pad>']
        self.unk_token_id = self.special_tokens['<unk>']
        self.bos_token_id = self.special_tokens['<bos>']
        self.eos_token_id = self.special_tokens['<eos>']
        self.mask_token_id = self.special_tokens['<mask>']
        
        # Biblical vocabulary seeds - important terms that should be preserved
        self.biblical_seeds = {
            # Names of God
            'God', 'LORD', 'Yahweh', 'Jehovah', 'Elohim', 'Adonai', 'El Shaddai',
            'Jesus', 'Christ', 'Messiah', 'Savior', 'Son of God', 'Holy Spirit',
            
            # Biblical books (abbreviated and full)
            'Genesis', 'Exodus', 'Leviticus', 'Numbers', 'Deuteronomy', 
            'Joshua', 'Judges', 'Ruth', 'Samuel', 'Kings', 'Chronicles',
            'Ezra', 'Nehemiah', 'Esther', 'Job', 'Psalms', 'Proverbs', 
            'Ecclesiastes', 'Song of Songs', 'Isaiah', 'Jeremiah', 'Lamentations',
            'Ezekiel', 'Daniel', 'Hosea', 'Joel', 'Amos', 'Obadiah', 'Jonah',
            'Micah', 'Nahum', 'Habakkuk', 'Zephaniah', 'Haggai', 'Zechariah',
            'Malachi', 'Matthew', 'Mark', 'Luke', 'John', 'Acts', 'Romans',
            'Corinthians', 'Galatians', 'Ephesians', 'Philippians', 'Colossians',
            'Thessalonians', 'Timothy', 'Titus', 'Philemon', 'Hebrews', 'James',
            'Peter', 'Jude', 'Revelation',
            
            # Key theological terms
            'salvation', 'redemption', 'sanctification', 'justification',
            'righteousness', 'holiness', 'grace', 'mercy', 'love', 'faith',
            'hope', 'prayer', 'worship', 'praise', 'glory', 'kingdom',
            'covenant', 'testament', 'gospel', 'disciple', 'apostle',
            'prophet', 'priest', 'temple', 'sacrifice', 'atonement',
            'resurrection', 'eternal', 'heaven', 'hell', 'judgment',
            'sin', 'forgiveness', 'repentance', 'baptism', 'communion',
            
            # Biblical characters
            'Adam', 'Eve', 'Noah', 'Abraham', 'Isaac', 'Jacob', 'Joseph',
            'Moses', 'Aaron', 'David', 'Solomon', 'Elijah', 'Elisha',
            'Isaiah', 'Jeremiah', 'Ezekiel', 'Daniel', 'Mary', 'Peter',
            'Paul', 'John', 'James', 'Andrew', 'Philip', 'Thomas'
        }
        
        # Regex patterns for preprocessing (using standard character classes instead of \p{})
        self.pre_tokenize_pattern = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+""",
            re.IGNORECASE
        )
        
    def train(self, texts: List[str], save_path: Optional[str] = None) -> None:
        """Train the BPE tokenizer on the provided texts."""
        print(f"Training BPE tokenizer with vocab_size={self.vocab_size}")
        
        # Pre-tokenize all texts
        word_freqs = defaultdict(int)
        for text in texts:
            words = self._pre_tokenize(text)
            for word in words:
                word_freqs[word] += 1
        
        print(f"Found {len(word_freqs)} unique words")
        
        # Initialize alphabet
        alphabet = set()
        for word in word_freqs:
            alphabet.update(word)
        alphabet = sorted(list(alphabet))
        
        print(f"Alphabet size: {len(alphabet)}")
        
        # Initialize vocabulary with special tokens and alphabet
        vocab = self.special_tokens.copy()
        next_id = len(self.special_tokens)
        
        for char in alphabet:
            if char not in vocab:
                vocab[char] = next_id
                next_id += 1
        
        # Add biblical seed terms as single tokens
        for seed in self.biblical_seeds:
            if seed not in vocab:
                vocab[seed] = next_id
                next_id += 1
        
        # Split words into characters for initial BPE
        splits = {}
        for word, freq in word_freqs.items():
            splits[word] = list(word)
        
        # Learn BPE merges
        merges = []
        while len(vocab) < self.vocab_size:
            # Count pairs
            pairs = defaultdict(int)
            for word, freq in word_freqs.items():
                symbols = splits[word]
                for i in range(len(symbols) - 1):
                    pairs[(symbols[i], symbols[i + 1])] += freq
            
            if not pairs:
                break
                
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            # Merge the pair
            new_token = best_pair[0] + best_pair[1]
            vocab[new_token] = next_id
            next_id += 1
            merges.append(best_pair)
            
            # Update splits
            for word in word_freqs:
                symbols = splits[word]
                new_symbols = []
                i = 0
                while i < len(symbols):
                    if (i < len(symbols) - 1 and 
                        symbols[i] == best_pair[0] and 
                        symbols[i + 1] == best_pair[1]):
                        new_symbols.append(new_token)
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                splits[word] = new_symbols
            
            if len(vocab) % 1000 == 0:
                print(f"Vocabulary size: {len(vocab)}")
        
        self.vocab = vocab
        self.merges = merges
        self.bpe_ranks = {merge: i for i, merge in enumerate(merges)}
        
        # Create reverse vocabulary
        self.id_to_token = {v: k for k, v in vocab.items()}
        
        print(f"Training complete. Final vocabulary size: {len(vocab)}")
        
        if save_path:
            self.save(save_path)
    
    def _pre_tokenize(self, text: str) -> List[str]:
        """Pre-tokenize text into words."""
        # Basic whitespace and punctuation tokenization
        # This is simplified - in practice you'd want more sophisticated pre-tokenization
        text = text.strip()
        if not text:
            return []
        
        # Split on whitespace but preserve biblical terms
        words = []
        tokens = text.split()
        
        for token in tokens:
            # Check if token contains biblical seed terms
            found_seed = False
            for seed in self.biblical_seeds:
                if seed.lower() in token.lower():
                    # Preserve the seed as a unit
                    parts = token.lower().split(seed.lower())
                    if len(parts) > 1:
                        for i, part in enumerate(parts[:-1]):
                            if part:
                                words.append(part)
                            words.append(seed)
                        if parts[-1]:
                            words.append(parts[-1])
                        found_seed = True
                        break
            
            if not found_seed:
                words.append(token)
        
        return words
    
    def _bpe_encode(self, word: str) -> List[str]:
        """Apply BPE encoding to a single word."""
        if not word:
            return []
        
        # Check if word is a biblical seed term
        if word in self.biblical_seeds and word in self.vocab:
            return [word]
        
        # Split word into characters
        word_tokens = list(word)
        
        # Apply merges
        pairs = self._get_pairs(word_tokens)
        
        if not pairs:
            return word_tokens
        
        while True:
            # Find the highest priority pair to merge
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            
            if bigram not in self.bpe_ranks:
                break
                
            # Merge the pair
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word_tokens):
                try:
                    j = word_tokens.index(first, i)
                    new_word.extend(word_tokens[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word_tokens[i:])
                    break
                
                if (i < len(word_tokens) - 1 and 
                    word_tokens[i] == first and 
                    word_tokens[i + 1] == second):
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word_tokens[i])
                    i += 1
            
            word_tokens = new_word
            if len(word_tokens) == 1:
                break
            
            pairs = self._get_pairs(word_tokens)
        
        return word_tokens
    
    def _get_pairs(self, word_tokens: List[str]) -> Set[Tuple[str, str]]:
        """Get all adjacent pairs in the word."""
        pairs = set()
        prev_char = word_tokens[0]
        for char in word_tokens[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        if not text.strip():
            return [self.pad_token_id] if add_special_tokens else []
        
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.bos_token_id)
        
        # Pre-tokenize
        words = self._pre_tokenize(text)
        
        # BPE encode each word
        for word in words:
            if not word.strip():
                continue
                
            bpe_tokens = self._bpe_encode(word)
            for token in bpe_tokens:
                token_id = self.vocab.get(token, self.unk_token_id)
                token_ids.append(token_id)
        
        if add_special_tokens:
            token_ids.append(self.eos_token_id)
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                
                if skip_special_tokens and token in self.special_tokens:
                    continue
                    
                tokens.append(token)
            else:
                if not skip_special_tokens:
                    tokens.append('<unk>')
        
        # Join tokens and clean up
        text = ''.join(tokens)
        
        # Basic cleanup - add spaces around words
        # This is simplified - real implementation would be more sophisticated
        text = re.sub(r'([a-zA-Z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'(\w)([^\w\s])', r'\1 \2', text)
        text = re.sub(r'([^\w\s])(\w)', r'\1 \2', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)
    
    def save(self, path: str) -> None:
        """Save tokenizer to file."""
        save_data = {
            'vocab': self.vocab,
            'merges': self.merges,
            'bpe_ranks': self.bpe_ranks,
            'special_tokens': self.special_tokens,
            'biblical_seeds': self.biblical_seeds,
            'vocab_size': self.vocab_size
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Tokenizer saved to {path}")
    
    def load(self, path: str) -> None:
        """Load tokenizer from file."""
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.vocab = save_data['vocab']
        self.merges = save_data['merges']
        self.bpe_ranks = save_data['bpe_ranks']
        self.special_tokens = save_data['special_tokens']
        self.biblical_seeds = save_data['biblical_seeds']
        self.vocab_size = save_data['vocab_size']
        
        # Rebuild reverse mapping
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        # Update special token IDs
        self.pad_token_id = self.special_tokens['<pad>']
        self.unk_token_id = self.special_tokens['<unk>']
        self.bos_token_id = self.special_tokens['<bos>']
        self.eos_token_id = self.special_tokens['<eos>']
        self.mask_token_id = self.special_tokens['<mask>']
        
        print(f"Tokenizer loaded from {path}")
    
    def get_vocab(self) -> Dict[str, int]:
        """Return the vocabulary dictionary."""
        return self.vocab.copy()
    
    def token_to_id(self, token: str) -> int:
        """Convert token to ID."""
        return self.vocab.get(token, self.unk_token_id)
    
    def id_to_token(self, token_id: int) -> str:
        """Convert ID to token."""
        return self.id_to_token.get(token_id, '<unk>')


def train_biblical_tokenizer(texts: List[str], vocab_size: int = 32000, save_path: str = None) -> BiblicalBPETokenizer:
    """
    Train a BiblicalBPETokenizer on the provided texts.
    
    Args:
        texts: List of training texts
        vocab_size: Target vocabulary size
        save_path: Path to save trained tokenizer
        
    Returns:
        Trained tokenizer
    """
    tokenizer = BiblicalBPETokenizer(vocab_size=vocab_size)
    tokenizer.train(texts, save_path=save_path)
    return tokenizer


if __name__ == "__main__":
    # Example usage and testing
    sample_texts = [
        "In the beginning God created the heavens and the earth.",
        "For God so loved the world that he gave his one and only Son.",
        "The LORD is my shepherd, I lack nothing.",
        "Jesus said to them, 'I am the bread of life.'",
        "Trust in the LORD with all your heart and lean not on your own understanding."
    ]
    
    print("Training sample biblical tokenizer...")
    tokenizer = train_biblical_tokenizer(sample_texts, vocab_size=1000)
    
    # Test encoding/decoding
    test_text = "Jesus Christ is the Son of God and our Savior."
    print(f"\nTest text: {test_text}")
    
    encoded = tokenizer.encode(test_text)
    print(f"Encoded: {encoded}")
    
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")
    
    print(f"\nVocabulary size: {len(tokenizer)}")
    print("Sample tokens:", list(tokenizer.vocab.keys())[:20])