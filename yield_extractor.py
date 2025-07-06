#!/usr/bin/env python3
"""
Bank of Jamaica Financial Data Retrieval Script
Extracts treasury bill and money market security data from BOJ website
"""

import os
import json
import requests
from bs4 import BeautifulSoup
import logging
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import base64
from io import BytesIO
from PIL import Image
import fitz  # PyMuPDF for PDF to image conversion
from openai import OpenAI
from typing import List, Dict, Optional, Tuple
import time
import re
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


# Optional: Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip

# Configuration
BASE_URL = "https://boj.org.jm"
TREASURY_BILLS_URL = f"{BASE_URL}/category/notices/treasury-bills/"
MONEY_MARKET_URL = f"{BASE_URL}/category/notices/money-market/"
IMAGES_DIR = Path("images")
LOGS_DIR = Path("logs")
MODEL = "gpt-4.1-mini"  # Updated to available model

# Create directories
IMAGES_DIR.mkdir(exist_ok=True)
(IMAGES_DIR / "treasury_bills").mkdir(exist_ok=True)
(IMAGES_DIR / "money_market").mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'boj_extractor.log'),
        logging.StreamHandler()
    ]
)

class BOJExtractor:
    """
    Bank of Jamaica financial data extractor
    
    Performance Notes:
    - Current implementation is sequential for simplicity
    - Can be parallelized using asyncio/aiohttp for 5-10x speed improvement
    - Consider ThreadPoolExecutor for PDF processing and concurrent LLM calls
    """
    def __init__(self, max_early_date: Optional[str] = None):
        """
        Initialize the BOJ extractor
        
        Args:
            max_early_date: Maximum early date in YYYY-MM-DD format (optional)
        """
        # Get OpenAI API key from environment variable
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
            
        self.client = OpenAI(api_key=openai_api_key)
        self.max_early_date = datetime.strptime(max_early_date, "%Y-%m-%d") if max_early_date else None
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        # Load processed URLs log
        self.processed_urls_file = LOGS_DIR / 'processed_urls.json'
        self.processed_urls = self._load_processed_urls()
        
    def _load_processed_urls(self) -> set:
        """Load previously processed PDF URLs"""
        if self.processed_urls_file.exists():
            with open(self.processed_urls_file, 'r') as f:
                return set(json.load(f))
        return set()
    
    def _save_processed_urls(self):
        """Save processed PDF URLs to log file"""
        with open(self.processed_urls_file, 'w') as f:
            json.dump(list(self.processed_urls), f, indent=2)
    
    def _should_process_by_date(self, title: str, url: str) -> bool:
        """
        Use LLM to determine if a document should be processed based on date
        """
        if not self.max_early_date:
            return True
            
        schema = {
            "type": "object",
            "properties": {
                "should_process": {
                    "type": "boolean",
                    "description": "Whether the document date is on or after the maximum early date"
                },
                "extracted_date": {
                    "type": "string",
                    "description": "The date extracted from the title in YYYY-MM-DD format"
                }
            },
            "required": ["should_process", "extracted_date"],
            "additionalProperties": False
        }
        
        messages = [
            {
                "role": "system",
                "content": f"You are a date extraction specialist. Given a document title and URL, determine if the document date is on or after {self.max_early_date.strftime('%Y-%m-%d')}. Extract the date from the title and provide a binary decision."
            },
            {
                "role": "user",
                "content": f"Document title: {title}\nDocument URL: {url}\nMaximum early date: {self.max_early_date.strftime('%Y-%m-%d')}\n\nShould this document be processed?"
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "date_filter",
                        "strict": True,
                        "schema": schema
                    }
                }
            )
            result = json.loads(response.choices[0].message.content)
            logging.info(f"Date filtering for '{title}': {result}")
            return result["should_process"]
        except Exception as e:
            logging.error(f"Error in date filtering: {e}")
            return True  # Default to processing if error occurs

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((requests.RequestException, requests.Timeout))
    )
    def _get_page_content(self, url: str) -> Optional[BeautifulSoup]:
        """Get page content with error handling"""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            logging.error(f"Error fetching {url}: {e}")
            return None

    def _find_pdf_url(self, notice_url: str) -> Optional[str]:
        """Find PDF URL on a notice page"""
        soup = self._get_page_content(notice_url)
        if not soup:
            return None
            
        # Look for PDF links
        pdf_links = soup.find_all('a', href=re.compile(r'\.pdf$', re.IGNORECASE))
        if pdf_links:
            pdf_url = pdf_links[0]['href']
            if not pdf_url.startswith('http'):
                pdf_url = BASE_URL + pdf_url
            return pdf_url
            
        # Also check for any links containing "pdf" or "download"
        download_links = soup.find_all('a', href=re.compile(r'(pdf|download)', re.IGNORECASE))
        if download_links:
            pdf_url = download_links[0]['href']
            if not pdf_url.startswith('http'):
                pdf_url = BASE_URL + pdf_url
            return pdf_url
            
        return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((requests.RequestException, requests.Timeout))
    )
    def _download_and_convert_pdf(self, pdf_url: str, data_type: str = "general") -> Optional[str]:
        """Download PDF and convert to image with data type separation"""
        if pdf_url in self.processed_urls:
            logging.info(f"PDF already processed: {pdf_url}")
            return None
            
        try:
            # Download PDF
            response = self.session.get(pdf_url, timeout=60)
            response.raise_for_status()
            
            # Create data type specific directory
            type_dir = IMAGES_DIR / data_type
            type_dir.mkdir(exist_ok=True)
            
            # Create filename based on URL hash
            url_hash = hashlib.md5(pdf_url.encode()).hexdigest()[:8]
            image_filename = f"pdf_{url_hash}.png"
            image_path = type_dir / image_filename
            
            # Convert PDF to image using PyMuPDF
            pdf_document = fitz.open(stream=response.content, filetype="pdf")
            page = pdf_document[0]  # Get first page
            
            # Render page to image
            mat = fitz.Matrix(2, 2)  # Increase resolution
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Save image
            with open(image_path, 'wb') as f:
                f.write(img_data)
                
            pdf_document.close()
            
            # Add to processed URLs
            self.processed_urls.add(pdf_url)
            self._save_processed_urls()
            
            logging.info(f"Converted PDF to image: {image_path}")
            return str(image_path)
            
        except Exception as e:
            logging.error(f"Error processing PDF {pdf_url}: {e}")
            return None

    def _extract_treasury_bill_data(self, image_path: str) -> Optional[Dict]:
        """Extract treasury bill data using GPT-4.1"""
        schema = {
            "type": "object",
            "properties": {
                "bills": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tenure": {
                                "type": "number",
                                "description": "Tenure in months from issue to maturity. Can be 3, 6, 9, or 12 months"
                            },
                            "yield": {
                                "type": "number",
                                "description": "Average yield percentage with 2 decimal precision"
                            },
                            "month": {
                                "type": "string",
                                "description": "Month name from issue date",
                                "enum": ["January", "February", "March", "April", "May", "June", 
                                       "July", "August", "September", "October", "November", "December"]
                            },
                            "year": {
                                "type": "number",
                                "description": "Year from issue date"
                            }
                        },
                        "required": ["tenure", "yield", "month", "year"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["bills"],
            "additionalProperties": False
        }
        
        # Encode image to base64
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        messages = [
            {
                "role": "system", 
                "content": "You are a financial data extraction specialist analyzing Government of Jamaica Treasury Bill auction results. From the provided auction result document, extract data for ALL treasury bills listed. For each bill, determine: 1) Tenure in months (calculate from issue date to maturity date, round to nearest 3,6,9,12), 2) Average yield as percentage with 2 decimal precision, 3) Month name from issue date, 4) Year from issue date. Return data for every bill in the document."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract treasury bill data from Government of Jamaica auction results including tenure, yield, month and year for each bill"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_data}"
                        }
                    }
                ]
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "treasury_bills",
                        "strict": True,
                        "schema": schema
                    }
                }
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logging.error(f"Error extracting treasury bill data: {e}")
            return None

    def _extract_money_market_data(self, image_path: str) -> Optional[Dict]:
        """Extract money market security data using GPT-4.1"""
        schema = {
            "type": "object",
            "properties": {
                "securities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "security_type": {
                                "type": "string",
                                "description": "Type of security (e.g., FR BIN)"
                            },
                            "fixed_rate": {
                                "type": "number",
                                "description": "Fixed rate percentage with 2 decimal precision"
                            },
                            "years_to_maturity": {
                                "type": "number",
                                "description": "Years until maturity"
                            },
                            "yield": {
                                "type": "number",
                                "description": "Yield percentage with 2 decimal precision"
                            },
                            "issue_date": {
                                "type": "string",
                                "description": "Issue date in YYYY-MM-DD format"
                            },
                            "maturity_date": {
                                "type": "string",
                                "description": "Maturity date in YYYY-MM-DD format"
                            }
                        },
                        "required": ["security_type", "fixed_rate", "years_to_maturity", "yield", "issue_date", "maturity_date"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["securities"],
            "additionalProperties": False
        }
        
        # Encode image to base64
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        messages = [
            {
                "role": "system",
                "content": "You are a financial data extraction specialist analyzing Government of Jamaica money market security auction results. Extract data for ALL securities listed including security type, fixed rate, years to maturity, yield, issue date, and maturity date."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract money market security data from this auction results document"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_data}"
                        }
                    }
                ]
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "money_market_securities",
                        "strict": True,
                        "schema": schema
                    }
                }
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logging.error(f"Error extracting money market data: {e}")
            return None

    def _filter_money_market_notices(self, notices: List[Dict]) -> List[Dict]:
        """Filter money market notices using LLM"""
        schema = {
            "type": "object",
            "properties": {
                "valid_notices": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"},
                            "title": {"type": "string"},
                            "reason": {"type": "string"}
                        },
                        "required": ["url", "title", "reason"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["valid_notices"],
            "additionalProperties": False
        }
        
        notices_text = "\n".join([f"Title: {n['title']}\nURL: {n['url']}" for n in notices])
        
        messages = [
            {
                "role": "system",
                "content": "You are a financial document filter. Filter notices based on these criteria: 1) Results only (no announcements), 2) Long-term securities (no 14 or 30-day auctions, just multi-year notes). Return only valid URLs that meet both criteria. Note that `Results` will be clearly stated in the title for all valid options."
            },
            {
                "role": "user",
                "content": f"Filter these money market notices:\n\n{notices_text}"
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "filtered_notices",
                        "strict": True,
                        "schema": schema
                    }
                }
            )
            result = json.loads(response.choices[0].message.content)
            return result["valid_notices"]
        except Exception as e:
            logging.error(f"Error filtering notices: {e}")
            return notices

    def extract_treasury_bills(self) -> List[Dict]:
        """extract treasury bill data"""
        logging.info("Starting treasury bill scraping...")
        all_data = []
        page = 1
        
        while True:
            if page == 1:
                url = TREASURY_BILLS_URL
            else:
                url = f"{TREASURY_BILLS_URL}page/{page}/"
            
            logging.info(f"Scraping treasury bills page {page}: {url}")
            soup = self._get_page_content(url)
            
            if not soup:
                break
                
            # Find auction result links and apply date filtering
            auction_links = []
            date_limit_reached = False
            articles = soup.find_all('article')
            
            for article in articles:
                title_element = article.find('h3') or article.find('h2') or article.find('a')
                link_element = article.find('a', href=re.compile(r'results-of-auction'))
                
                if title_element and link_element:
                    title = title_element.get_text(strip=True)
                    href = link_element['href']
                    
                    if not href.startswith('http'):
                        href = BASE_URL + href
                    
                    # Check if we should process based on date
                    if self._should_process_by_date(title, href):
                        auction_links.append((title, href))
                    else:
                        logging.info(f"Reached date limit with: {title}")
                        date_limit_reached = True
                        break  # Stop checking articles on this page
                        
            # Break pagination if we hit the date limit
            if date_limit_reached:
                logging.info(f"Stopping pagination - reached date limit on page {page}")
                break
                        
            if not auction_links:
                logging.info(f"No auction links found on page {page}")
                if not articles:  # Only break if page is completely empty
                    break
                else:
                    page += 1
                    continue  # Keep going to next page
                
            # Process each auction link
            processed_count = 0
            for title, notice_url in auction_links:
                logging.info(f"Processing: {title}")
                
                # Find PDF URL
                pdf_url = self._find_pdf_url(notice_url)
                if not pdf_url:
                    logging.warning(f"No PDF found for: {notice_url}")
                    continue
                    
                # Download and convert PDF
                image_path = self._download_and_convert_pdf(pdf_url, "treasury_bills")
                if not image_path:
                    continue
                    
                # Extract data
                data = self._extract_treasury_bill_data(image_path)
                if data:
                    data['source_title'] = title
                    data['source_url'] = notice_url
                    data['pdf_url'] = pdf_url
                    data['image_path'] = image_path
                    all_data.append(data)
                    processed_count += 1
                    
                # Optional: Add small delay if needed
                # time.sleep(0.5)
            
            # Log summary for this page
            if processed_count == 0:
                logging.info(f"No documents processed on page {page} (no valid auction links)")
            else:
                logging.info(f"Processed {processed_count} documents on page {page}")
                
            page += 1
            
        logging.info(f"Treasury bill scraping completed. Found {len(all_data)} documents.")
        return all_data

    def extract_money_market(self) -> List[Dict]:
        """extract money market security data"""
        logging.info("Starting money market scraping...")
        all_data = []
        page = 1
        
        while True:
            if page == 1:
                url = MONEY_MARKET_URL
            else:
                url = f"{MONEY_MARKET_URL}page/{page}/"
            
            logging.info(f"Scraping money market page {page}: {url}")
            soup = self._get_page_content(url)
            
            if not soup:
                break
                
            # Find all notice links
            notices = []
            articles = soup.find_all('article')
            
            for article in articles:
                title_element = article.find('h3') or article.find('h2')
                link_element = article.find('a')
                
                if title_element and link_element:
                    title = title_element.get_text(strip=True)
                    href = link_element['href']
                    
                    if not href.startswith('http'):
                        href = BASE_URL + href
                    
                    notices.append({'title': title, 'url': href})
                    
            if not notices:
                logging.info(f"No notices found on page {page}")
                break
                
            # Filter notices using LLM
            valid_notices = self._filter_money_market_notices(notices)
            logging.info(f"Filtered {len(notices)} notices to {len(valid_notices)} valid ones")
            
            # Process each valid notice and check for date limit
            date_limit_reached = False
            processed_count = 0
            
            for notice in valid_notices:
                title = notice['title']
                notice_url = notice['url']
                
                # Check date filtering
                if not self._should_process_by_date(title, notice_url):
                    logging.info(f"Reached date limit with: {title}")
                    date_limit_reached = True
                    break  # Stop processing this page
                    
                logging.info(f"Processing: {title}")
                
                # Find PDF URL
                pdf_url = self._find_pdf_url(notice_url)
                if not pdf_url:
                    logging.warning(f"No PDF found for: {notice_url}")
                    continue
                    
                # Download and convert PDF
                image_path = self._download_and_convert_pdf(pdf_url, "money_market")
                if not image_path:
                    continue
                    
                # Extract data
                data = self._extract_money_market_data(image_path)
                if data:
                    data['source_title'] = title
                    data['source_url'] = notice_url
                    data['pdf_url'] = pdf_url
                    data['image_path'] = image_path
                    all_data.append(data)
                    processed_count += 1
                    
                # Optional: Add small delay if needed  
                # time.sleep(0.5)
                
            # Break pagination if we hit the date limit
            if date_limit_reached:
                logging.info(f"Stopping pagination - reached date limit on page {page}")
                break
                
            # Log summary for this page
            if processed_count == 0:
                logging.info(f"No documents processed on page {page} (filtered out or no valid articles)")
            else:
                logging.info(f"Processed {processed_count} documents on page {page}")
                
            page += 1
            
        logging.info(f"Money market scraping completed. Found {len(all_data)} documents.")
        return all_data

    def _save_results(self, data: Dict, data_type: str) -> str:
        """Save results to separate files by data type"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = LOGS_DIR / f"boj_{data_type}_{timestamp}.json"
        
        # Add metadata
        output_data = {
            **data,
            'timestamp': datetime.now().isoformat(),
            'max_early_date': self.max_early_date.isoformat() if self.max_early_date else None,
            'data_type': data_type
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
            
        logging.info(f"{data_type.title()} results saved to: {output_file}")
        return str(output_file)

    def run_full_extract(self) -> Dict:
        """Run full scraping for both treasury bills and money market"""
        logging.info("Starting full BOJ data scraping...")
        
        # extract both data types
        treasury_data = self.extract_treasury_bills()
        money_market_data = self.extract_money_market()
        
        # Save to separate files
        treasury_file = self._save_results({'treasury_bills': treasury_data}, 'treasury_bills')
        money_market_file = self._save_results({'money_market': money_market_data}, 'money_market')
        
        results = {
            'treasury_bills': treasury_data,
            'money_market': money_market_data,
            'files': {
                'treasury_bills': treasury_file,
                'money_market': money_market_file
            }
        }
        
        logging.info(f"Scraping completed.")
        logging.info(f"Treasury bills: {len(treasury_data)} documents -> {treasury_file}")
        logging.info(f"Money market: {len(money_market_data)} documents -> {money_market_file}")
        
        return results


def main():
    """Main function to run the extractor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='BOJ Financial Data Extractor')
    parser.add_argument('--max-early-date', help='Maximum early date (YYYY-MM-DD)')
    parser.add_argument('--treasury-only', action='store_true', help='Only extract treasury bills')
    parser.add_argument('--money-market-only', action='store_true', help='Only extract money market')
    
    args = parser.parse_args()
    
    extractor = BOJExtractor(args.max_early_date)
    
    if args.treasury_only:
        treasury_data = extractor.extract_treasury_bills()
        output_file = extractor._save_results({'treasury_bills': treasury_data}, 'treasury_bills')
        results = {'treasury_bills': treasury_data, 'output_file': output_file}
    elif args.money_market_only:
        money_market_data = extractor.extract_money_market()
        output_file = extractor._save_results({'money_market': money_market_data}, 'money_market')
        results = {'money_market': money_market_data, 'output_file': output_file}
    else:
        results = extractor.run_full_extract()
    
    print(f"Scraping completed!")
    if 'files' in results:
        print(f"Treasury bills file: {results['files']['treasury_bills']}")
        print(f"Money market file: {results['files']['money_market']}")
    elif 'output_file' in results:
        print(f"Output file: {results['output_file']}")
    
    # Print summary without file paths (too verbose)
    summary = {k: v for k, v in results.items() if k not in ['files', 'output_file']}
    for key, value in summary.items():
        if isinstance(value, list):
            print(f"{key}: {len(value)} documents")
        elif isinstance(value, dict) and 'bills' in value:
            total_bills = sum(len(doc.get('bills', [])) for doc in value if isinstance(doc, dict))
            print(f"{key}: {len(value)} documents, {total_bills} total bills/securities")


if __name__ == "__main__":
    main()