#!/usr/bin/env python3
"""
BOJ Data Processor - Converts latest JSON outputs to CSV
Processes treasury bills and money market securities data
"""

import os
import json
import pandas as pd
import logging
from datetime import datetime, timedelta
from pathlib import Path
import glob
import re
from typing import List, Dict, Optional, Tuple
import calendar

# Configuration
LOGS_DIR = Path("logs")
OUTPUT_DIR = Path("output")

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(OUTPUT_DIR / 'boj_processor.log'),
        logging.StreamHandler()
    ]
)

class BOJProcessor:
    def __init__(self):
        """Initialize the BOJ data processor"""
        if not LOGS_DIR.exists():
            raise ValueError(f"Logs directory {LOGS_DIR} not found")
    
    def _find_latest_files(self) -> Tuple[Optional[str], Optional[str]]:
        """Find the latest treasury bills and money market JSON files"""
        treasury_pattern = str(LOGS_DIR / "boj_treasury_bills_*.json")
        money_market_pattern = str(LOGS_DIR / "boj_money_market_*.json")
        
        treasury_files = glob.glob(treasury_pattern)
        money_market_files = glob.glob(money_market_pattern)
        
        # Sort by filename (which includes timestamp) to get latest
        latest_treasury = max(treasury_files) if treasury_files else None
        latest_money_market = max(money_market_files) if money_market_files else None
        
        logging.info(f"Latest treasury file: {latest_treasury}")
        logging.info(f"Latest money market file: {latest_money_market}")
        
        return latest_treasury, latest_money_market
    
    def _calculate_dates_from_treasury_bill(self, bill: Dict, source_info: Dict) -> Tuple[str, str]:
        """Calculate issue date and maturity date for treasury bills"""
        # Extract date info from bill
        month_name = bill['month']
        year = bill['year']
        tenure_months = bill['tenure']
        
        # Convert month name to number
        month_num = list(calendar.month_name).index(month_name)
        
        # Create issue date (assume first day of month for simplicity)
        issue_date = datetime(year, month_num, 1)
        
        # Calculate maturity date
        maturity_date = issue_date + timedelta(days=tenure_months * 30.44)  # Average month length
        
        return issue_date.strftime('%Y-%m-%d'), maturity_date.strftime('%Y-%m-%d')
    
    def _calculate_time_to_maturity(self, maturity_date_str: str) -> float:
        """Calculate time to maturity in years from today"""
        maturity_date = datetime.strptime(maturity_date_str, '%Y-%m-%d')
        today = datetime.now()
        days_to_maturity = (maturity_date - today).days
        return round(days_to_maturity / 365.25, 2)  # Convert to years
    
    def _process_treasury_bills(self, file_path: str) -> List[Dict]:
        """Process treasury bills data and return latest 3, 6, 9 month securities"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        processed_bills = []
        
        for document in data.get('treasury_bills', []):
            source_url = document.get('source_url', '')
            source_title = document.get('source_title', '')
            
            for bill in document.get('bills', []):
                # Only process 3, 6, 9 month tenures
                if bill['tenure'] not in [3, 6, 9]:
                    continue
                
                # Calculate dates
                issue_date, maturity_date = self._calculate_dates_from_treasury_bill(bill, document)
                time_to_maturity = self._calculate_time_to_maturity(maturity_date)
                
                processed_bills.append({
                    'security_type': 'Treasury Bill',
                    'tenure_months': bill['tenure'],
                    'yield': bill['yield'],
                    'issue_date': issue_date,
                    'maturity_date': maturity_date,
                    'time_to_maturity_years': time_to_maturity,
                    'source_url': source_url,
                    'source_title': source_title
                })
        
        # Get latest bill for each tenure (3, 6, 9 months)
        df = pd.DataFrame(processed_bills)
        if df.empty:
            return []
        
        # Sort by issue date and get the latest for each tenure
        latest_bills = []
        for tenure in [3, 6, 9]:
            tenure_bills = df[df['tenure_months'] == tenure]
            if not tenure_bills.empty:
                latest_bill = tenure_bills.loc[tenure_bills['issue_date'].idxmax()]
                latest_bills.append(latest_bill.to_dict())
        
        logging.info(f"Processed {len(latest_bills)} latest treasury bills (3, 6, 9 month)")
        return latest_bills
    
    def _process_money_market(self, file_path: str) -> List[Dict]:
        """Process money market data and deduplicate by maturity date"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        processed_securities = []
        
        for document in data.get('money_market', []):
            source_url = document.get('source_url', '')
            source_title = document.get('source_title', '')
            
            for security in document.get('securities', []):
                time_to_maturity = self._calculate_time_to_maturity(security['maturity_date'])
                
                processed_securities.append({
                    'security_type': security.get('security_type', 'Money Market'),
                    'fixed_rate': security.get('fixed_rate'),
                    'yield': security['yield'],
                    'issue_date': security['issue_date'],
                    'maturity_date': security['maturity_date'],
                    'time_to_maturity_years': time_to_maturity,
                    'years_to_maturity': security.get('years_to_maturity'),
                    'source_url': source_url,
                    'source_title': source_title
                })
        
        # Deduplicate by maturity date, keeping the one with latest issue date
        # Deduplicate by maturity date proximity (within 3 months)
        df = pd.DataFrame(processed_securities)
        if df.empty:
            return []
        
        df['maturity_date_dt'] = pd.to_datetime(df['maturity_date'])
        df['issue_date_dt'] = pd.to_datetime(df['issue_date'])
        df = df.sort_values('maturity_date_dt')
        
        final_securities = []
        used_indices = set()
        
        for i, row in df.iterrows():
            if i in used_indices:
                continue
                
            # Find securities within 3 months (90 days)
            close_securities = []
            for j, other_row in df.iterrows():
                if j in used_indices:
                    continue
                days_diff = abs((row['maturity_date_dt'] - other_row['maturity_date_dt']).days)
                if days_diff <= 90:
                    close_securities.append(j)
            
            # Keep the one with latest issue date
            if close_securities:
                latest_idx = df.loc[close_securities]['issue_date_dt'].idxmax()
                security = df.loc[latest_idx].to_dict()
                security.pop('maturity_date_dt')
                security.pop('issue_date_dt')
                final_securities.append(security)
                used_indices.update(close_securities)
        
        logging.info(f"Processed {len(final_securities)} unique money market securities")
        return final_securities
    
    def _create_combined_csv(self, treasury_bills: List[Dict], money_market: List[Dict]) -> str:
        """Create combined CSV with both treasury bills and money market data"""
        all_securities = []
        
        # Add treasury bills
        for bill in treasury_bills:
            all_securities.append({
                'security_type': bill['security_type'],
                'tenure_months': bill.get('tenure_months'),
                'fixed_rate': None,  # Treasury bills don't have fixed rates
                'yield': bill['yield'],
                'issue_date': bill['issue_date'],
                'maturity_date': bill['maturity_date'],
                'time_to_maturity_years': bill['time_to_maturity_years'],
                'years_to_maturity': bill.get('tenure_months', 0) / 12 if bill.get('tenure_months') else None,
                'source_url': bill['source_url'],
                'source_title': bill['source_title']
            })
        
        # Add money market securities
        for security in money_market:
            all_securities.append({
                'security_type': security['security_type'],
                'tenure_months': None,  # Money market securities use years_to_maturity
                'fixed_rate': security.get('fixed_rate'),
                'yield': security['yield'],
                'issue_date': security['issue_date'],
                'maturity_date': security['maturity_date'],
                'time_to_maturity_years': security['time_to_maturity_years'],
                'years_to_maturity': security.get('years_to_maturity'),
                'source_url': security['source_url'],
                'source_title': security['source_title']
            })
        
        # Create DataFrame
        df = pd.DataFrame(all_securities)
        
        # Sort by time to maturity
        df = df.sort_values('time_to_maturity_years')
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = OUTPUT_DIR / f"boj_securities_{timestamp}.csv"
        
        # Save to CSV
        df.to_csv(csv_filename, index=False)
        
        logging.info(f"Combined CSV saved: {csv_filename}")
        logging.info(f"Total securities: {len(all_securities)} ({len(treasury_bills)} treasury bills, {len(money_market)} money market)")
        
        return str(csv_filename)
    
    def _create_separate_csvs(self, treasury_bills: List[Dict], money_market: List[Dict]) -> Tuple[str, str]:
        """Create separate CSV files for treasury bills and money market"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Treasury Bills CSV
        treasury_csv = None
        if treasury_bills:
            df_treasury = pd.DataFrame(treasury_bills)
            df_treasury = df_treasury.sort_values('tenure_months')
            treasury_csv = OUTPUT_DIR / f"boj_treasury_bills_{timestamp}.csv"
            df_treasury.to_csv(treasury_csv, index=False)
            logging.info(f"Treasury bills CSV saved: {treasury_csv}")
        
        # Money Market CSV
        money_market_csv = None
        if money_market:
            df_money_market = pd.DataFrame(money_market)
            df_money_market = df_money_market.sort_values('time_to_maturity_years')
            money_market_csv = OUTPUT_DIR / f"boj_money_market_{timestamp}.csv"
            df_money_market.to_csv(money_market_csv, index=False)
            logging.info(f"Money market CSV saved: {money_market_csv}")
        
        return str(treasury_csv) if treasury_csv else None, str(money_market_csv) if money_market_csv else None
    
    def process_latest_data(self, output_format: str = 'combined') -> Dict[str, str]:
        """
        Process the latest JSON files and create CSV output
        
        Args:
            output_format: 'combined', 'separate', or 'both'
        
        Returns:
            Dictionary with file paths
        """
        logging.info("Starting BOJ data processing...")
        
        # Find latest files
        treasury_file, money_market_file = self._find_latest_files()
        
        if not treasury_file and not money_market_file:
            raise ValueError("No JSON files found in logs directory")
        
        # Process data
        treasury_bills = []
        money_market = []
        
        if treasury_file:
            treasury_bills = self._process_treasury_bills(treasury_file)
        
        if money_market_file:
            money_market = self._process_money_market(money_market_file)
        
        # Create output files
        results = {}
        
        if output_format in ['combined', 'both']:
            combined_csv = self._create_combined_csv(treasury_bills, money_market)
            results['combined'] = combined_csv
        
        if output_format in ['separate', 'both']:
            treasury_csv, money_market_csv = self._create_separate_csvs(treasury_bills, money_market)
            if treasury_csv:
                results['treasury_bills'] = treasury_csv
            if money_market_csv:
                results['money_market'] = money_market_csv
        
        logging.info("BOJ data processing completed!")
        return results


def main():
    """Main function to run the processor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='BOJ Data Processor - Convert JSON to CSV')
    parser.add_argument('--format', choices=['combined', 'separate', 'both'], 
                       default='combined', help='Output format')
    parser.add_argument('--show-data', action='store_true', 
                       help='Display the data in console')
    
    args = parser.parse_args()
    
    try:
        processor = BOJProcessor()
        results = processor.process_latest_data(args.format)
        
        print("Processing completed!")
        for output_type, file_path in results.items():
            print(f"{output_type.title()} CSV: {file_path}")
            
        # Show data if requested
        if args.show_data and 'combined' in results:
            print("\n--- Data Preview ---")
            df = pd.read_csv(results['combined'])
            print(df.to_string(index=False))
            
    except Exception as e:
        logging.error(f"Processing failed: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()