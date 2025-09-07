#!/usr/bin/env python3
"""
Comprehensive D.L. Moody Dataset Collector

Combines ALL URLs from all previous collector scripts:
- master_dataset_collector.py
- enhanced_moody_collector.py  
- verified_moody_collector.py

Generates both .txt and .json files matching exact MVLM training format.
Compatible with existing training scripts without modification.

Author: Manus AI - Comprehensive URL Collection
"""

import os
import sys
import requests
import time
import re
import json
import hashlib
from pathlib import Path
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComprehensiveMoodyCollector:
    """Comprehensive collector using ALL URLs from all previous scripts."""
    
    def __init__(self):
        # Find the dataset directory relative to script location
        script_dir = Path(__file__).parent
        self.dataset_dir = script_dir / "mvlm_training_dataset_complete" / "mvlm_comprehensive_dataset" / "biblical_classical"
        
        # Create session for requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Statistics
        self.stats = {
            'total_works': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'total_words': 0,
            'total_characters': 0,
            'by_source': {}
        }
        
        # Ensure directories exist
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories."""
        # Create subcategories for Moody works
        subcategories = ['historical_biblical', 'contemporary_biblical']
        
        for subcat in subcategories:
            subcat_dir = self.dataset_dir / subcat
            subcat_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {subcat_dir}")
    
    def get_comprehensive_works_catalog(self) -> List[Dict]:
        """Get ALL URLs from ALL previous collector scripts."""
        
        works = [
            # FROM MASTER_DATASET_COLLECTOR.PY - Project Gutenberg URLs
            {
                'title': 'Men of the Bible',
                'url': 'https://www.gutenberg.org/files/24/24-0.txt',
                'category': 'historical_biblical',
                'source': 'Project Gutenberg',
                'type': 'book',
                'quality_score': 9.5,
                'biblical_alignment': 10.0
            },
            {
                'title': 'Sowing and Reaping',
                'url': 'https://www.gutenberg.org/files/30768/30768-0.txt',
                'category': 'historical_biblical',
                'source': 'Project Gutenberg',
                'type': 'book',
                'quality_score': 9.2,
                'biblical_alignment': 9.8
            },
            {
                'title': 'Sovereign Grace: Its Source, Its Nature and Its Effects',
                'url': 'https://www.gutenberg.org/files/22/22-0.txt',
                'category': 'historical_biblical',
                'source': 'Project Gutenberg',
                'type': 'book',
                'quality_score': 9.7,
                'biblical_alignment': 10.0
            },
            {
                'title': 'Weighed and Wanting: Addresses on the Ten Commandments',
                'url': 'https://www.gutenberg.org/files/29/29-0.txt',
                'category': 'historical_biblical',
                'source': 'Project Gutenberg',
                'type': 'book',
                'quality_score': 9.4,
                'biblical_alignment': 9.9
            },
            {
                'title': 'Secret Power; or, The Secret of Success in Christian Life and Work',
                'url': 'https://www.gutenberg.org/files/33341/33341-0.txt',
                'category': 'contemporary_biblical',
                'source': 'Project Gutenberg',
                'type': 'book',
                'quality_score': 9.3,
                'biblical_alignment': 9.7
            },
            {
                'title': 'The Way to God and How to Find It',
                'url': 'https://www.gutenberg.org/files/21/21-0.txt',
                'category': 'contemporary_biblical',
                'source': 'Project Gutenberg',
                'type': 'book',
                'quality_score': 9.6,
                'biblical_alignment': 10.0
            },
            {
                'title': 'Pleasure & Profit in Bible Study',
                'url': 'https://www.gutenberg.org/files/18/18-0.txt',
                'category': 'historical_biblical',
                'source': 'Project Gutenberg',
                'type': 'book',
                'quality_score': 9.1,
                'biblical_alignment': 9.8
            },
            {
                'title': 'To The Work! To The Work! Exhortations to Christians',
                'url': 'https://www.gutenberg.org/files/26/26-0.txt',
                'category': 'contemporary_biblical',
                'source': 'Project Gutenberg',
                'type': 'book',
                'quality_score': 9.0,
                'biblical_alignment': 9.6
            },
            {
                'title': 'The Overcoming Life, and Other Sermons',
                'url': 'https://www.gutenberg.org/files/33015/33015-0.txt',
                'category': 'contemporary_biblical',
                'source': 'Project Gutenberg',
                'type': 'book',
                'quality_score': 9.4,
                'biblical_alignment': 9.8
            },
            {
                'title': 'Wondrous Love, and other Gospel addresses',
                'url': 'https://www.gutenberg.org/files/27/27-0.txt',
                'category': 'contemporary_biblical',
                'source': 'Project Gutenberg',
                'type': 'book',
                'quality_score': 9.2,
                'biblical_alignment': 9.7
            },
            {
                'title': 'That Gospel Sermon on the Blessed Hope',
                'url': 'https://www.gutenberg.org/files/28/28-0.txt',
                'category': 'contemporary_biblical',
                'source': 'Project Gutenberg',
                'type': 'book',
                'quality_score': 9.3,
                'biblical_alignment': 9.9
            },
            {
                'title': 'Prevailing Prayer: What Hinders It?',
                'url': 'https://www.gutenberg.org/files/20/20-0.txt',
                'category': 'contemporary_biblical',
                'source': 'Project Gutenberg',
                'type': 'book',
                'quality_score': 9.1,
                'biblical_alignment': 9.8
            },
            {
                'title': 'Thoughts for the Quiet Hour',
                'url': 'https://www.gutenberg.org/files/37292/37292-0.txt',
                'category': 'contemporary_biblical',
                'source': 'Project Gutenberg',
                'type': 'book',
                'quality_score': 8.9,
                'biblical_alignment': 9.5
            },
            {
                'title': "Moody's Stories: Being a Second Volume of Anecdotes, Incidents, and Illustrations",
                'url': 'https://www.gutenberg.org/files/33024/33024-0.txt',
                'category': 'contemporary_biblical',
                'source': 'Project Gutenberg',
                'type': 'book',
                'quality_score': 9.0,
                'biblical_alignment': 9.4
            },
            {
                'title': "Moody's Anecdotes And Illustrations",
                'url': 'https://www.gutenberg.org/files/15/15-0.txt',
                'category': 'contemporary_biblical',
                'source': 'Project Gutenberg',
                'type': 'book',
                'quality_score': 9.0,
                'biblical_alignment': 9.4
            },
            {
                'title': 'Bible Characters',
                'url': 'https://www.gutenberg.org/files/33023/33023-0.txt',
                'category': 'historical_biblical',
                'source': 'Project Gutenberg',
                'type': 'book',
                'quality_score': 9.5,
                'biblical_alignment': 10.0
            },
            
            # FROM ENHANCED_MOODY_COLLECTOR.PY - Internet Archive URLs
            {
                'title': 'The Overcoming Life (Archive)',
                'url': 'https://archive.org/stream/overcominglife00mood/overcominglife00mood_djvu.txt',
                'category': 'contemporary_biblical',
                'source': 'Internet Archive',
                'type': 'book',
                'quality_score': 9.2,
                'biblical_alignment': 9.7
            },
            {
                'title': 'Heaven: Its Hope, Its Inhabitants, Its Music, Its Service',
                'url': 'https://archive.org/stream/heavenitshopeits00mood/heavenitshopeits00mood_djvu.txt',
                'category': 'historical_biblical',
                'source': 'Internet Archive',
                'type': 'book',
                'quality_score': 9.4,
                'biblical_alignment': 9.8
            },
            {
                'title': 'Bible Characters (Archive)',
                'url': 'https://archive.org/stream/biblecharacters00mood/biblecharacters00mood_djvu.txt',
                'category': 'historical_biblical',
                'source': 'Internet Archive',
                'type': 'book',
                'quality_score': 9.3,
                'biblical_alignment': 9.9
            },
            {
                'title': 'Secret Power (Archive)',
                'url': 'https://archive.org/stream/secretpower00mood/secretpower00mood_djvu.txt',
                'category': 'contemporary_biblical',
                'source': 'Internet Archive',
                'type': 'book',
                'quality_score': 9.1,
                'biblical_alignment': 9.6
            },
            {
                'title': 'Sowing and Reaping (Archive)',
                'url': 'https://archive.org/stream/sowingandreaping00mood/sowingandreaping00mood_djvu.txt',
                'category': 'contemporary_biblical',
                'source': 'Internet Archive',
                'type': 'book',
                'quality_score': 9.0,
                'biblical_alignment': 9.5
            },
            {
                'title': 'Weighed and Wanting (Archive)',
                'url': 'https://archive.org/stream/weighedandwantin00mood/weighedandwantin00mood_djvu.txt',
                'category': 'historical_biblical',
                'source': 'Internet Archive',
                'type': 'book',
                'quality_score': 9.2,
                'biblical_alignment': 9.8
            },
            {
                'title': 'Thoughts for the Quiet Hour (Archive)',
                'url': 'https://archive.org/stream/thoughtsforquiet00mood/thoughtsforquiet00mood_djvu.txt',
                'category': 'contemporary_biblical',
                'source': 'Internet Archive',
                'type': 'book',
                'quality_score': 8.8,
                'biblical_alignment': 9.3
            },
            {
                'title': "Moody's Stories (Archive)",
                'url': 'https://archive.org/stream/moodystories00mood/moodystories00mood_djvu.txt',
                'category': 'contemporary_biblical',
                'source': 'Internet Archive',
                'type': 'book',
                'quality_score': 8.9,
                'biblical_alignment': 9.2
            },
            {
                'title': "Moody's Latest Sermons",
                'url': 'https://archive.org/stream/moodyslatestser00mood/moodyslatestser00mood_djvu.txt',
                'category': 'contemporary_biblical',
                'source': 'Internet Archive',
                'type': 'book',
                'quality_score': 9.1,
                'biblical_alignment': 9.6
            },
            
            # FROM ENHANCED_MOODY_COLLECTOR.PY - CCEL URLs
            {
                'title': 'How to Study the Bible',
                'url': 'https://www.ccel.org/ccel/moody/how_to_study.txt',
                'category': 'historical_biblical',
                'source': 'CCEL',
                'type': 'book',
                'quality_score': 9.3,
                'biblical_alignment': 9.8
            },
            {
                'title': 'The Gospel Awakening',
                'url': 'https://www.ccel.org/ccel/moody/gospel_awakening.txt',
                'category': 'contemporary_biblical',
                'source': 'CCEL',
                'type': 'book',
                'quality_score': 9.2,
                'biblical_alignment': 9.7
            },
            
            # FROM VERIFIED_MOODY_COLLECTOR.PY - Bible Believers Sermons
            {
                'title': 'The Qualifications for Soul Winning',
                'url': 'https://www.biblebelievers.com/moody_sermons/moody_01.html',
                'category': 'contemporary_biblical',
                'source': 'Bible Believers',
                'type': 'sermon',
                'quality_score': 9.4,
                'biblical_alignment': 9.8
            },
            {
                'title': 'Christ All in All',
                'url': 'https://www.biblebelievers.com/moody_sermons/moody_02.html',
                'category': 'contemporary_biblical',
                'source': 'Bible Believers',
                'type': 'sermon',
                'quality_score': 9.5,
                'biblical_alignment': 9.9
            },
            {
                'title': 'Does God Answer Prayer?',
                'url': 'https://www.biblebelievers.com/moody_sermons/moody_03.html',
                'category': 'contemporary_biblical',
                'source': 'Bible Believers',
                'type': 'sermon',
                'quality_score': 9.2,
                'biblical_alignment': 9.7
            },
            {
                'title': 'The Blood',
                'url': 'https://www.biblebelievers.com/moody_sermons/moody_04.html',
                'category': 'historical_biblical',
                'source': 'Bible Believers',
                'type': 'sermon',
                'quality_score': 9.6,
                'biblical_alignment': 10.0
            },
            {
                'title': 'Repentance',
                'url': 'https://www.biblebelievers.com/moody_sermons/moody_05.html',
                'category': 'contemporary_biblical',
                'source': 'Bible Believers',
                'type': 'sermon',
                'quality_score': 9.3,
                'biblical_alignment': 9.8
            },
            {
                'title': 'Nicodemus',
                'url': 'https://www.biblebelievers.com/moody_sermons/moody_06.html',
                'category': 'historical_biblical',
                'source': 'Bible Believers',
                'type': 'sermon',
                'quality_score': 9.4,
                'biblical_alignment': 9.9
            },
            {
                'title': 'Zacchaeus',
                'url': 'https://www.biblebelievers.com/moody_sermons/moody_07.html',
                'category': 'historical_biblical',
                'source': 'Bible Believers',
                'type': 'sermon',
                'quality_score': 9.2,
                'biblical_alignment': 9.7
            },
            {
                'title': 'The Prodigal Son',
                'url': 'https://www.biblebelievers.com/moody_sermons/moody_08.html',
                'category': 'historical_biblical',
                'source': 'Bible Believers',
                'type': 'sermon',
                'quality_score': 9.5,
                'biblical_alignment': 9.8
            },
            {
                'title': 'The Rich Young Ruler',
                'url': 'https://www.biblebelievers.com/moody_sermons/moody_09.html',
                'category': 'historical_biblical',
                'source': 'Bible Believers',
                'type': 'sermon',
                'quality_score': 9.3,
                'biblical_alignment': 9.7
            },
            {
                'title': 'The Woman at the Well',
                'url': 'https://www.biblebelievers.com/moody_sermons/moody_10.html',
                'category': 'historical_biblical',
                'source': 'Bible Believers',
                'type': 'sermon',
                'quality_score': 9.4,
                'biblical_alignment': 9.8
            }
        ]
        
        return works
    
    def clean_text_content(self, text: str, source: str, title: str) -> str:
        """Clean text based on source type."""
        
        if source == 'Project Gutenberg':
            # Remove Project Gutenberg headers and footers
            start_patterns = [
                r'\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*',
                r'START OF (?:THE|THIS) PROJECT GUTENBERG.*?EBOOK.*?\*\*\*'
            ]
            
            for pattern in start_patterns:
                start_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if start_match:
                    text = text[start_match.end():]
                    break
            
            end_patterns = [
                r'\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*',
                r'END OF (?:THE|THIS) PROJECT GUTENBERG.*?EBOOK.*?\*\*\*'
            ]
            
            for pattern in end_patterns:
                end_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if end_match:
                    text = text[:end_match.start()]
                    break
        
        elif source == 'Bible Believers':
            # Extract text from HTML
            try:
                soup = BeautifulSoup(text, 'html.parser')
                
                # Remove navigation, ads, etc.
                for element in soup(['nav', 'header', 'footer', 'aside', 'script', 'style']):
                    element.decompose()
                
                # Get main content
                content = soup.get_text()
                text = content
            except Exception as e:
                logger.warning(f"HTML parsing failed for {title}: {e}")
        
        elif source == 'Internet Archive':
            # Clean Internet Archive OCR text
            text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)  # Remove page numbers
            text = re.sub(r'\n{3,}', '\n\n', text)  # Reduce excessive line breaks
        
        # General cleaning
        text = re.sub(r'\r\n', '\n', text)  # Normalize line endings
        text = re.sub(r'[ \t]+', ' ', text)  # Normalize spaces
        text = text.strip()
        
        return text
    
    def generate_hash(self, content: str) -> str:
        """Generate hash for content."""
        return hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
    
    def create_safe_filename(self, title: str) -> str:
        """Create safe filename from title."""
        # Remove special characters and normalize
        safe_title = re.sub(r'[^\w\s-]', '', title)
        safe_title = re.sub(r'[-\s]+', '-', safe_title)
        safe_title = safe_title.strip('-').lower()
        return safe_title
    
    def download_work(self, work: Dict) -> Optional[Tuple[str, str]]:
        """Download a single work and return (text_content, json_metadata)."""
        
        title = work['title']
        url = work['url']
        
        logger.info(f"Downloading: {title}")
        logger.info(f"URL: {url}")
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Get content
            raw_content = response.text
            
            # Clean content
            clean_content = self.clean_text_content(raw_content, work['source'], title)
            
            if len(clean_content.strip()) < 1000:  # Minimum content threshold
                logger.warning(f"Content too short for {title}: {len(clean_content)} chars")
                return None
            
            # Create metadata header for text file
            metadata_header = f"""Title: {title}
Author: D.L. Moody (1837-1899)
Source: {work['source']}
Category: {work['category']}
Type: {work['type']}
Quality Score: {work['quality_score']}
Biblical Alignment: {work['biblical_alignment']}
Added to Dataset: {time.strftime('%Y-%m-%d')}

---

"""
            
            # Combine header and content
            final_text = metadata_header + clean_content
            
            # Create JSON metadata
            content_hash = self.generate_hash(final_text)
            word_count = len(final_text.split())
            
            json_metadata = {
                "title": title,
                "author": "D.L. Moody",
                "source": work['source'],
                "category": "biblical_classical",
                "subcategory": work['category'],
                "quality_score": work['quality_score'],
                "biblical_alignment": work['biblical_alignment'],
                "word_count": word_count,
                "url": url,
                "hash": content_hash,
                "priority": 1,
                "filepath": ""  # Will be set when saving
            }
            
            logger.info(f"Successfully processed: {title} ({word_count:,} words)")
            
            # Update stats
            self.stats['total_words'] += word_count
            self.stats['total_characters'] += len(final_text)
            
            if work['source'] not in self.stats['by_source']:
                self.stats['by_source'][work['source']] = 0
            self.stats['by_source'][work['source']] += 1
            
            # Respectful delay
            time.sleep(2)
            
            return final_text, json_metadata
            
        except Exception as e:
            logger.error(f"Failed to download {title}: {e}")
            return None
    
    def save_work(self, text_content: str, json_metadata: Dict, work: Dict) -> bool:
        """Save work as both .txt and .json files."""
        
        try:
            # Create safe filename
            safe_title = self.create_safe_filename(work['title'])
            content_hash = json_metadata['hash']
            base_filename = f"{safe_title}_{content_hash}"
            
            # Determine subcategory directory
            subcat_dir = self.dataset_dir / work['category']
            
            # File paths
            txt_path = subcat_dir / f"{base_filename}.txt"
            json_path = subcat_dir / f"{base_filename}.json"
            
            # Update filepath in metadata
            json_metadata['filepath'] = str(txt_path.relative_to(self.dataset_dir.parent.parent.parent))
            
            # Save text file
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            
            # Save JSON file
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved: {txt_path}")
            logger.info(f"Saved: {json_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save {work['title']}: {e}")
            return False
    
    def collect_all_works(self):
        """Main collection function."""
        
        works = self.get_comprehensive_works_catalog()
        self.stats['total_works'] = len(works)
        
        logger.info(f"Starting comprehensive D.L. Moody collection")
        logger.info(f"Total works to attempt: {len(works)}")
        logger.info(f"Target directory: {self.dataset_dir}")
        
        successful_works = []
        failed_works = []
        
        for work in works:
            logger.info(f"\n--- Processing: {work['title']} ---")
            
            # Download work
            result = self.download_work(work)
            
            if result:
                text_content, json_metadata = result
                
                # Save work
                if self.save_work(text_content, json_metadata, work):
                    successful_works.append(work['title'])
                    self.stats['successful_downloads'] += 1
                else:
                    failed_works.append(work['title'])
                    self.stats['failed_downloads'] += 1
            else:
                failed_works.append(work['title'])
                self.stats['failed_downloads'] += 1
        
        # Generate final report
        self.generate_final_report(successful_works, failed_works)
    
    def generate_final_report(self, successful: List[str], failed: List[str]):
        """Generate final collection report."""
        
        logger.info("\n" + "="*60)
        logger.info("D.L. MOODY COLLECTION COMPLETE")
        logger.info("="*60)
        logger.info(f"Total works attempted: {self.stats['total_works']}")
        logger.info(f"Successful downloads: {self.stats['successful_downloads']}")
        logger.info(f"Failed downloads: {self.stats['failed_downloads']}")
        logger.info(f"Success rate: {self.stats['successful_downloads']/self.stats['total_works']*100:.1f}%")
        logger.info(f"Total words collected: {self.stats['total_words']:,}")
        logger.info(f"Total characters: {self.stats['total_characters']:,}")
        
        logger.info("\nBy Source:")
        for source, count in self.stats['by_source'].items():
            logger.info(f"  {source}: {count} works")
        
        if successful:
            logger.info(f"\nSuccessful Downloads ({len(successful)}):")
            for title in successful:
                logger.info(f"  ✓ {title}")
        
        if failed:
            logger.info(f"\nFailed Downloads ({len(failed)}):")
            for title in failed:
                logger.info(f"  ✗ {title}")
        
        logger.info(f"\nDataset Location: {self.dataset_dir}")
        logger.info("Ready for MVLM training!")

def main():
    """Main execution function."""
    
    print("D.L. Moody Comprehensive Dataset Collector")
    print("==========================================")
    print("Collecting from ALL previous collector scripts...")
    print()
    
    try:
        collector = ComprehensiveMoodyCollector()
        collector.collect_all_works()
        
    except KeyboardInterrupt:
        print("\nCollection interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Collection failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

