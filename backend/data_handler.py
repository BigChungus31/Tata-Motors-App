# NEW data_handler.py

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from functools import lru_cache
import logging
import os
from dotenv import load_dotenv
from supabase import create_client, Client
from utils import DataLoadError, setup_web_logging, ProgressTracker, clean_json_response

class DataLoader:
    """Optimized Supabase data loading and column detection"""
    
    def __init__(self):
        load_dotenv()
        self.logger = setup_web_logging(__name__)
        self.progress = ProgressTracker()
        # Cache for repeated data loads
        self._data_cache = {}
        
        # Initialize Supabase client - try both naming conventions
        self.supabase_url = (
            os.getenv('NEXT_PUBLIC_SUPABASE_URL') or 
            os.getenv('SUPABASE_URL')
        )
        self.supabase_key = (
            os.getenv('NEXT_PUBLIC_SUPABASE_ANON_KEY') or 
            os.getenv('SUPABASE_ANON_KEY')
        )
        
        if not self.supabase_url or not self.supabase_key:
            raise DataLoadError(
                f"Supabase credentials not found in environment variables. "
                f"URL: {'Found' if self.supabase_url else 'Missing'}, "
                f"Key: {'Found' if self.supabase_key else 'Missing'}"
            )
        
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
    
    def list_tables(self):
        """Debug function to list all available tables"""
        try:
            # This is a workaround to see table structure
            response = self.supabase.rpc('get_tables').execute()
            return response.data
        except Exception as e:
            print(f"Could not list tables: {str(e)}")

    def load_supabase_data(self, depot_name: str = 'Hassanpur') -> pd.DataFrame:
        table_name = 'HassanpurTataMotors'

        """Load data from Supabase table"""
        try:
            cache_key = f"supabase_{table_name}"
            
            if cache_key in self._data_cache:
                self.logger.info("Using cached Supabase data")
                return self._data_cache[cache_key].copy()
            
            self.progress.start(4, "Loading data from Supabase")
            
            # Load all data with proper pagination
            all_data = []
            page_size = 1000
            start = 0
            
            while True:
                end = start + page_size - 1
                response = self.supabase.table(table_name).select("*").range(start, end).execute()
                
                if not response.data:
                    break
                    
                all_data.extend(response.data)
                print(f"Loaded {len(response.data)} records, total: {len(all_data)}")
                
                # If we got fewer records than page_size, we're done
                if len(response.data) < page_size:
                    break
                    
                start += page_size

            print(f"Final total records loaded: {len(all_data)}")
            
            if not all_data:
                raise DataLoadError(f"No data found in table: {table_name}")

            self.progress.update(1, "Data fetched from Supabase")
            
            # Convert to DataFrame
            raw_data = pd.DataFrame(all_data)
            
            self.progress.update(2, "Data converted to DataFrame")
            
            # Memory optimization - convert object columns to category where appropriate
            for col in raw_data.select_dtypes(include=['object']).columns:
                if raw_data[col].nunique() / len(raw_data) < 0.5:  # Less than 50% unique values
                    raw_data[col] = raw_data[col].astype('category')
            
            self.progress.update(3, "Memory optimized")
            
            # Cache the result (limit cache size to prevent memory issues)
            if len(self._data_cache) < 3:  # Limit cache size
                self._data_cache[cache_key] = raw_data.copy()
            
            self.logger.info(f"Loaded data with shape: {raw_data.shape}, Memory usage optimized")
            self.progress.update(4, "Data loading completed")
            
            return raw_data
            
        except Exception as e:
            raise DataLoadError(f"Error loading data from Supabase: {str(e)}")
    
    def filter_by_depot(self, data: pd.DataFrame, depot_name: str) -> pd.DataFrame:
        """Filter data by depot name"""
        if 'Name 1' in data.columns:
            filtered_data = data[data['Name 1'].str.contains(depot_name, case=False, na=False)]
            self.logger.info(f"Filtered data for {depot_name}: {filtered_data.shape}")
            return filtered_data
        return data

    @lru_cache(maxsize=32)
    def detect_columns(self, columns_tuple: tuple) -> Dict[str, Optional[str]]:
        """Optimized column detection with caching"""
        # Convert tuple back to list for processing
        columns = list(columns_tuple)
        
        # Optimized column mappings with priority order
        column_mappings = {
            'date': [
                'Document Date', 'Posting Date', 'Document_Date', 'Posting_Date',
                'document_date', 'posting_date', 'transaction_date'
            ],
            'quantity': [
                'Quantity', 'Qty', 'Qty in unit of entry', 'Consumption_Qty',
                'quantity', 'qty', 'consumption_qty'
            ],
            'material': [
                'Material', 'Material Number', 'Item', 'Part', 'Product',
                'material', 'material_number', 'item', 'part'
            ],
            'description': [
                'Material Description', 'Item Description', 
                'Product Description', 'Description',
                'material_description', 'item_description', 'description'
            ],
            'amount': [
                'Amount', 'Price', 'Total Amount', 'Total Price',
                'amount', 'price', 'total_amount', 'total_price'
            ]
        }
        
        detected_columns = {}
        
        # Create lowercase column lookup for faster matching
        col_lookup = {col.lower(): col for col in columns}
        
        for col_type, possible_names in column_mappings.items():
            detected_columns[col_type] = None
            
            # Optimized exact match first
            for name in possible_names:
                if name.lower() in col_lookup:
                    detected_columns[col_type] = col_lookup[name.lower()]
                    break
            
            # Optimized partial match if no exact match
            if detected_columns[col_type] is None:
                keywords = col_type.lower().split()
                for col in columns:
                    col_lower = col.lower()
                    if any(keyword in col_lower for keyword in keywords):
                        detected_columns[col_type] = col
                        break
        
        return detected_columns
    
    def detect_columns_from_df(self, df: pd.DataFrame) -> Dict[str, Optional[str]]:
        """Wrapper to use cached column detection"""
        # Convert columns to tuple for caching
        columns_tuple = tuple(df.columns)
        return self.detect_columns(columns_tuple)
    
    def clear_cache(self):
        """Clear data cache to free memory"""
        self._data_cache.clear()
        self.detect_columns.cache_clear()

# Rest of the DataValidator class remains the same
class DataValidator:
    """Optimized data validation and cleaning"""
    
    def __init__(self):
        self.logger = setup_web_logging()
        self.progress = ProgressTracker()
        # Cache for processed data
        self._processing_cache = {}
    
    def process_raw_data(self, raw_data: pd.DataFrame, detected_columns: Dict[str, str]) -> pd.DataFrame:
        """Optimized data processing with vectorized operations"""
        try:
            # Create cache key based on data shape and columns
            cache_key = f"{raw_data.shape}_{hash(str(detected_columns))}"
            
            if cache_key in self._processing_cache:
                self.logger.info("Using cached processed data")
                return self._processing_cache[cache_key].copy()
            
            self.progress.start(6, "Processing raw data...")
            
            # Check required columns exist
            required_cols = ['date', 'quantity', 'material', 'amount']
            missing_cols = [col for col in required_cols if not detected_columns.get(col)]
            
            if missing_cols:
                raise DataLoadError(f"Missing required columns: {missing_cols}")
            
            # Work on a copy to avoid modifying original
            data = raw_data.copy()
            
            # Optimized date processing - vectorized with multiple format attempts
            date_col = detected_columns['date']
            self.progress.update(1, "Processing dates...")
            
            # Try multiple date formats efficiently
            data['document_date'] = pd.NaT  # Use NaT instead of None for datetime

            # Convert the date column to string first to handle mixed types
            date_series = data[date_col].astype(str)

            # Try multiple date formats efficiently
            date_formats = ['%d-%m-%Y', '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y']

            for fmt in date_formats:
                mask = data['document_date'].isna()
                if mask.any():
                    try:
                        data.loc[mask, 'document_date'] = pd.to_datetime(
                            date_series[mask], 
                            format=fmt, 
                            errors='coerce'
                        )
                    except:
                        continue
            
            # Final attempt with infer format
            mask = data['document_date'].isna()
            if mask.any():
                data.loc[mask, 'document_date'] = pd.to_datetime(
                    data.loc[mask, date_col], 
                    errors='coerce',
                    infer_datetime_format=True
                )
            
            self.progress.update(2, "Dates processed")
            
            # Optimized quantity processing - vectorized
            qty_col = detected_columns['quantity']
            data['consumption_qty'] = pd.to_numeric(data[qty_col], errors='coerce').abs()
            self.progress.update(3, "Quantities processed")
            
            # Optimized amount processing - vectorized
            amount_col = detected_columns['amount']
            data['amount'] = pd.to_numeric(data[amount_col], errors='coerce').abs()
            
            # Calculate unit price (amount per unit quantity)
            data['unit_price'] = np.where(
                data['consumption_qty'] > 0,
                data['amount'] / data['consumption_qty'],
                0
            )
            
            # Optimized material processing
            material_col = detected_columns['material']
            data['material_number'] = data[material_col].astype(str)
            self.progress.update(4, "Materials processed")
            
            # Optimized description processing
            if detected_columns.get('description'):
                data['material_description'] = data[detected_columns['description']].astype(str)
            else:
                data['material_description'] = data['material_number']
            self.progress.update(5, "Descriptions processed")
            
            if 'Name 1' in data.columns:
                data['depot_name'] = data['Name 1'].astype(str)

            # Optimized filtering - single boolean mask
            valid_mask = (
                (data['consumption_qty'] > 0) &
                (data['consumption_qty'].notna()) &
                (data['amount'] >= 0) &
                (data['amount'].notna()) &
                (data['material_number'].notna()) &
                (data['material_number'] != 'nan') &
                (data['document_date'].notna())
            )
            
            processed_data = data[valid_mask].copy()
            
            # Vectorized time feature creation
            processed_data['year'] = processed_data['document_date'].dt.year
            processed_data['month'] = processed_data['document_date'].dt.month
            processed_data['month_name'] = processed_data['document_date'].dt.month_name()
            processed_data['year_month'] = processed_data['document_date'].dt.to_period('M')
            
            # Memory optimization - convert to appropriate dtypes
            processed_data['year'] = processed_data['year'].astype('int16')
            processed_data['month'] = processed_data['month'].astype('int8')
            processed_data['month_name'] = processed_data['month_name'].astype('category')
            
            # Cache result (limit cache size)
            if len(self._processing_cache) < 2:
                self._processing_cache[cache_key] = processed_data.copy()
            
            self.progress.update(6, "Data validation completed")
            self.logger.info(f"Processed data shape: {processed_data.shape}")
            
            return processed_data
            
        except Exception as e:
            raise DataLoadError(f"Error processing data: {str(e)}")
    
    def identify_non_moving_materials(self, raw_data: pd.DataFrame, processed_data: pd.DataFrame) -> List[Dict]:
        """Optimized non-moving materials identification"""
        try:
            # Use sets for O(1) lookup instead of lists
            all_materials = set(raw_data['material_number'].dropna().astype(str).unique())
            active_materials = set(processed_data['material_number'].unique())
            non_moving = all_materials - active_materials
            
            if not non_moving:
                return []
            
            # Optimize description lookup - create lookup dict once
            desc_lookup = (
                raw_data[raw_data['material_number'].isin(non_moving)]
                .groupby('material_number')['material_description']
                .first()
                .to_dict()
            )
            
            # Vectorized creation of result list
            non_moving_list = [
                {
                    'material_number': material,
                    'description': desc_lookup.get(material, 'N/A')
                }
                for material in non_moving
                if material != 'nan'
            ]
            
            return non_moving_list
            
        except Exception as e:
            self.logger.error(f"Error identifying non-moving materials: {str(e)}")
            return []
    
    def detect_anomalies(self, processed_data: pd.DataFrame) -> Dict:
        """Optimized anomaly detection"""
        try:
            # Single pass statistics calculation
            qty_series = processed_data['consumption_qty']
            stats = qty_series.describe(percentiles=[0.99])
            
            mean_consumption = stats['mean']
            std_consumption = stats['std']
            q99 = stats['99%']
            
            threshold_high = max(q99, mean_consumption + 3 * std_consumption)
            
            # Vectorized filtering
            high_consumption_mask = qty_series > threshold_high
            high_consumption_records = processed_data[high_consumption_mask]
            
            anomalies = []
            if not high_consumption_records.empty:
                # Optimized aggregation
                material_stats = (
                    high_consumption_records
                    .groupby('material_number')
                    .agg({
                        'consumption_qty': ['count', 'max', 'mean'],
                        'material_description': 'first'
                    })
                    .round(2)
                )
                
                material_stats.columns = ['high_consumption_count', 'max_consumption', 'avg_high_consumption', 'description']
                
                # Vectorized anomaly creation
                anomalies = [
                    {
                        'material_number': material_number,
                        'description': row['description'],
                        'max_consumption': int(row['max_consumption']),
                        'avg_high_consumption': int(row['avg_high_consumption']),
                        'high_consumption_instances': int(row['high_consumption_count']),
                        'anomaly_type': 'Unusually High Consumption',
                        'threshold_used': int(threshold_high)
                    }
                    for material_number, row in material_stats.iterrows()
                ]
            
            return {
                'threshold_high': int(threshold_high),
                'q99_percentile': int(q99),
                'anomalies': anomalies,
                'total_anomalous_records': len(high_consumption_records),
                'statistics': {
                    'mean': int(mean_consumption),
                    'std': int(std_consumption),
                    'q99': int(q99)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {str(e)}")
            return {'anomalies': [], 'total_anomalous_records': 0}
    
    def clear_cache(self):
        """Clear processing cache"""
        self._processing_cache.clear()

# Utility function for memory monitoring
def get_memory_usage(df: pd.DataFrame) -> Dict:
    """Get memory usage statistics for a DataFrame"""
    memory_usage = df.memory_usage(deep=True)
    return {
        'total_mb': round(memory_usage.sum() / 1024 / 1024, 2),
        'per_column_mb': {col: round(usage / 1024 / 1024, 2) 
                         for col, usage in memory_usage.items()},
        'shape': df.shape,
        'dtypes': df.dtypes.to_dict()
    }

# Add a test function to verify the integration
if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    try:
        # Test DataLoader
        loader = DataLoader()
        print("✓ DataLoader initialized successfully")
        
        # Debug: Try to get table information
        try:
            # Try to get schema information
            schema_response = loader.supabase.rpc('get_schema').execute()
            print("Schema response:", schema_response.data)
        except Exception as e:
            print(f"Could not get schema: {str(e)}")
            
        # Try a simple query to see what happens
        try:
            # This should show us the actual error
            test_response = loader.supabase.from_('information_schema.tables').select('table_name').eq('table_schema', 'public').execute()
            print("Available tables:", test_response.data)
        except Exception as e:
            print(f"Could not list tables: {str(e)}")
        
        # Test DataValidator
        validator = DataValidator()
        print("✓ DataValidator initialized successfully")
        
        try:
            data = loader.load_supabase_data('Hassanpur')  # This will use HassanpurTataMotors
            print(f"✓ Data loaded successfully: {data.shape}")
        except Exception as e:
            print(f"Failed to load data: {str(e)}")
        
        if data is None:
            raise Exception("Could not load data with any table name variant")
        
        # Test column detection
        detected_cols = loader.detect_columns_from_df(data)
        print(f"✓ Column detection successful: {detected_cols}")
        
        # Test data processing
        processed_data = validator.process_raw_data(data, detected_cols)
        print(f"✓ Data processing successful: {processed_data.shape}")
        
        print("\nAll tests passed! Integration is working correctly.")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Make sure your .env file contains the correct Supabase credentials.")