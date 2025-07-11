#NEW Predictor.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

class SeasonalAnalyzer:
    def __init__(self, processed_data: pd.DataFrame):
        self.processed_data = processed_data
        self.monthly_consumption_matrix = None
        self.seasonal_patterns = None
        self.logger = logging.getLogger(__name__)
        # Add caching
        self._patterns_cache = None
        self._high_demand_cache = None

    def analyze_patterns(self) -> pd.DataFrame:
        """Optimized pattern analysis with caching"""
        if self._patterns_cache is not None:
            return self._patterns_cache
            
        # Single groupby operation instead of multiple
        monthly_patterns = self.processed_data.groupby([
            'material_number', 'month'
        ])['consumption_qty'].sum().unstack(fill_value=0)

        # Vectorized operations for better performance
        pattern_stats = pd.DataFrame(index=monthly_patterns.index)
        pattern_stats['avg_monthly_consumption'] = monthly_patterns.mean(axis=1).round(0).astype(int)
        pattern_stats['total_annual_consumption'] = monthly_patterns.sum(axis=1).round(0).astype(int)
        pattern_stats['consumption_std'] = monthly_patterns.std(axis=1).round(0).astype(int)
        
        # Avoid division by zero with vectorized operations
        pattern_stats['coefficient_of_variation'] = np.where(
            pattern_stats['avg_monthly_consumption'] > 0,
            (pattern_stats['consumption_std'] / pattern_stats['avg_monthly_consumption']).round(2),
            0
        )

        pattern_stats['peak_month'] = monthly_patterns.idxmax(axis=1)
        pattern_stats['peak_consumption'] = monthly_patterns.max(axis=1).round(0).astype(int)
        pattern_stats['low_month'] = monthly_patterns.idxmin(axis=1)
        pattern_stats['low_consumption'] = monthly_patterns.min(axis=1).round(0).astype(int)

        # Optimize month name mapping
        month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April',
                      5: 'May', 6: 'June', 7: 'July', 8: 'August',
                      9: 'September', 10: 'October', 11: 'November', 12: 'December'}

        pattern_stats['peak_month_name'] = pattern_stats['peak_month'].map(month_names)
        pattern_stats['low_month_name'] = pattern_stats['low_month'].map(month_names)
        
        # Handle inf values more efficiently
        pattern_stats = pattern_stats.replace([np.inf, -np.inf], 0).fillna(0)

        self.monthly_consumption_matrix = monthly_patterns
        self.seasonal_patterns = pattern_stats
        self._patterns_cache = pattern_stats
        return pattern_stats

    def get_high_demand_by_month(self) -> Dict[str, List[Dict]]:
        """Optimized high demand analysis with caching"""
        if self._high_demand_cache is not None:
            return self._high_demand_cache
            
        if self.monthly_consumption_matrix is None:
            self.analyze_patterns()
            
        month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April',
                      5: 'May', 6: 'June', 7: 'July', 8: 'August',
                      9: 'September', 10: 'October', 11: 'November', 12: 'December'}

        high_demand_by_month = {}
        
        # Process all months at once for better performance
        for month in range(1, 13):
            month_name = month_names[month]
            if month in self.monthly_consumption_matrix.columns:
                month_data = self.monthly_consumption_matrix[month]
                # Use quantile only once per month
                threshold = month_data.quantile(0.75)
                high_demand_materials = month_data[month_data >= threshold].sort_values(ascending=False)

                # Limit results to top 50 for performance
                high_demand_list = []
                for material_number, consumption in high_demand_materials.head(50).items():
                    if consumption > 0:
                        high_demand_list.append({
                            'material_number': material_number,
                            'consumption': int(consumption),
                            'recommended_stock': int(consumption * 1.1)
                        })
                high_demand_by_month[month_name] = high_demand_list
        
        self._high_demand_cache = high_demand_by_month
        return high_demand_by_month

    def clear_cache(self):
        """Clear cached results to free memory"""
        self._patterns_cache = None
        self._high_demand_cache = None

class MaterialClassifier:
    def __init__(self, seasonal_patterns: pd.DataFrame, material_descriptions: Dict[str, str] = None):
        self.seasonal_patterns = seasonal_patterns
        self.material_descriptions = material_descriptions or {}
        # Add caching
        self._classification_cache = None

    def classify_simplified(self) -> Dict[str, List[str]]:
        """Optimized classification with caching"""
        if self._classification_cache is not None:
            return self._classification_cache
            
        classifications = {
            'HD_Critical': [], 'HD_Variable': [], 'MD_Regular': [],
            'MD_Seasonal': [], 'LD_Stable': [], 'LD_Sporadic': []
        }

        # Calculate thresholds once
        high_threshold = self.seasonal_patterns['avg_monthly_consumption'].quantile(0.8)
        medium_threshold = self.seasonal_patterns['avg_monthly_consumption'].quantile(0.5)

        # Vectorized classification
        avg_consumption = self.seasonal_patterns['avg_monthly_consumption']
        cv = self.seasonal_patterns['coefficient_of_variation']
        
        # Create boolean masks for efficient classification
        high_demand_mask = avg_consumption >= high_threshold
        medium_demand_mask = (avg_consumption >= medium_threshold) & (avg_consumption < high_threshold)
        low_demand_mask = avg_consumption < medium_threshold
        high_cv_mask = cv >= 1.0

        # Classify materials using boolean indexing
        classifications['HD_Critical'] = list(self.seasonal_patterns[high_demand_mask & ~high_cv_mask].index)
        classifications['HD_Variable'] = list(self.seasonal_patterns[high_demand_mask & high_cv_mask].index)
        classifications['MD_Regular'] = list(self.seasonal_patterns[medium_demand_mask & ~high_cv_mask].index)
        classifications['MD_Seasonal'] = list(self.seasonal_patterns[medium_demand_mask & high_cv_mask].index)
        classifications['LD_Stable'] = list(self.seasonal_patterns[low_demand_mask & ~high_cv_mask].index)
        classifications['LD_Sporadic'] = list(self.seasonal_patterns[low_demand_mask & high_cv_mask].index)

        # Only create detailed classifications if material descriptions are available
        if self.material_descriptions:
            detailed_classifications = {}
            for category, materials in classifications.items():
                detailed_classifications[category] = []
                for material in materials:
                    if material in self.seasonal_patterns.index:
                        stats = self.seasonal_patterns.loc[material]
                        detailed_classifications[category].append({
                            'material_number': material,
                            'description': self.material_descriptions.get(material, 'N/A'),
                            'avg_consumption': int(stats['avg_monthly_consumption']),
                            'total_consumption': int(stats['total_annual_consumption']),
                            'coefficient_of_variation': round(stats['coefficient_of_variation'], 2)
                        })
            self._classification_cache = detailed_classifications
            return detailed_classifications
        
        self._classification_cache = classifications
        return classifications

    def clear_cache(self):
        """Clear cached results"""
        self._classification_cache = None

class InventoryPredictor:
    def __init__(self, seasonal_analyzer: SeasonalAnalyzer, material_descriptions: Dict[str, str]):
        self.seasonal_analyzer = seasonal_analyzer
        self.material_descriptions = material_descriptions
        self.logger = logging.getLogger(__name__)
        self._prediction_cache = {}
        # Add search cache for better performance
        self._search_cache = {}
        # Pre-compute month names dictionary
        self.month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April',
                           5: 'May', 6: 'June', 7: 'July', 8: 'August',
                           9: 'September', 10: 'October', 11: 'November', 12: 'December'}

    def search_materials(self, partial_description: str, limit: int = 20) -> List[Dict]:
        """Optimized material search with caching and limits"""
        if not partial_description or len(partial_description) < 2:
            return []
            
        # Use cache for repeated searches
        cache_key = f"{partial_description.lower()}_{limit}"
        if cache_key in self._search_cache:
            return self._search_cache[cache_key]
            
        partial_description = partial_description.lower().strip()
        matches = []
        
        # Early exit when limit is reached
        for material_number, description in self.material_descriptions.items():
            if partial_description in description.lower():
                matches.append({
                    'material_number': material_number,
                    'description': description
                })
                if len(matches) >= limit:
                    break

        # Optimized sorting
        matches.sort(key=lambda x: (
            not x['description'].lower().startswith(partial_description),
            len(x['description'])
        ))
        
        # Cache the result
        self._search_cache[cache_key] = matches
        return matches

    def predict_with_search(self, user_input: str, current_quantity: int = None) -> Dict:
        """Optimized prediction with search"""
        # Direct lookup optimization
        if user_input in self.material_descriptions:
            if current_quantity is None:
                return {
                    "type": "direct_match",
                    "material_number": user_input,
                    "description": self.material_descriptions[user_input],
                    "message": "Material found. Please provide current quantity for prediction."
                }
            else:
                return self.predict_status(user_input, current_quantity)

        # Limit search results for performance
        matches = self.search_materials(user_input, limit=10)
        if not matches:
            return {"type": "no_matches", "message": f"No materials found matching '{user_input}'"}

        if len(matches) == 1:
            material_number = matches[0]['material_number']
            if current_quantity is None:
                return {
                    "type": "single_match",
                    "material_number": material_number,
                    "description": matches[0]['description'],
                    "message": "Single material found. Please provide current quantity."
                }
            else:
                return self.predict_status(material_number, current_quantity)

        return {"type": "multiple_matches", "matches": matches, 
                "message": f"Found {len(matches)} matching materials."}

    def predict_status(self, material_number: str, current_quantity: int) -> Dict:
        """Optimized prediction with better caching"""
        # Enhanced cache key
        cache_key = f"{material_number}_{current_quantity}"
        if cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key]

        # Ensure patterns are available
        if self.seasonal_analyzer.seasonal_patterns is None:
            self.seasonal_analyzer.analyze_patterns()
            
        if material_number not in self.seasonal_analyzer.seasonal_patterns.index:
            return {"error": "Material number not found"}

        material_stats = self.seasonal_analyzer.seasonal_patterns.loc[material_number]
        material_desc = self.material_descriptions.get(material_number, "N/A")
        
        # Get monthly consumption data
        monthly_consumption = self.seasonal_analyzer.monthly_consumption_matrix.loc[material_number]
        avg_monthly_consumption = material_stats['avg_monthly_consumption']
        
        # Pre-calculate thresholds
        excess_threshold = avg_monthly_consumption * 1.1
        adequate_threshold = avg_monthly_consumption
        critical_threshold = avg_monthly_consumption * 0.5
        # Order quantity calculation
        target_qty = avg_monthly_consumption * 1.1
        if current_quantity >= avg_monthly_consumption:
            order_qty = 0
        else:
            order_qty = int(target_qty - current_quantity)

        # Determine status
        if current_quantity > excess_threshold:
            status = "Excess"
        elif current_quantity >= adequate_threshold:
            status = "Adequate"
        elif current_quantity >= critical_threshold:
            status = "Low"
        else:
            status = "Critical"

        # Calculate key metrics
        adequate_stock = int(avg_monthly_consumption)
        shortfall = max(0, adequate_stock - current_quantity)
        surplus = max(0, current_quantity - adequate_stock) if current_quantity > adequate_stock else None
        months_coverage = round(current_quantity / avg_monthly_consumption, 1) if avg_monthly_consumption > 0 else float('inf')

        # Optimize busy/slow months calculation
        monthly_consumption_filtered = monthly_consumption[monthly_consumption > 0]
        if len(monthly_consumption_filtered) >= 3:
            busy_months = monthly_consumption_filtered.nlargest(3)
            slow_months = monthly_consumption_filtered.nsmallest(3)
        else:
            busy_months = monthly_consumption_filtered
            slow_months = monthly_consumption_filtered

        # Build result dictionary
        result = {
            "type": "prediction",
            "material_number": material_number,
            "material_description": material_desc,
            "current_quantity": current_quantity,
            "status": status,
            "avg_monthly_consumption": int(avg_monthly_consumption),
            "months_coverage": months_coverage,
            "peak_consumption": int(material_stats['peak_consumption']),
            "peak_month": material_stats['peak_month_name'],
            "low_consumption": int(material_stats['low_consumption']),
            "low_month": material_stats['low_month_name'],
            "busy_months": {self.month_names[month]: int(consumption)
                           for month, consumption in busy_months.items()},
            "slow_months": {self.month_names[month]: int(consumption)
                           for month, consumption in slow_months.items()},
            "coefficient_of_variation": round(material_stats['coefficient_of_variation'], 2),
            "cv_interpretation": self._interpret_cv(material_stats['coefficient_of_variation']),
            "recommendation": self._generate_recommendation(status, material_stats, current_quantity),
            "order_quantity": order_qty,
            "demand_variability": f"CV: {material_stats['coefficient_of_variation']:.2f}",
        }

        # Add conditional fields
        if status in ["Low", "Critical"] and shortfall > 0:
            result["shortfall"] = shortfall
        elif surplus is not None and surplus > 0:
            result["surplus"] = surplus

        # Cache the result
        self._prediction_cache[cache_key] = result
        return result

    def _generate_recommendation(self, status: str, material_stats: pd.Series, current_quantity: int) -> str:
        """Optimized recommendation generation"""
        cv = material_stats['coefficient_of_variation']
        avg_consumption = material_stats['avg_monthly_consumption']
        optimal_stock = int(avg_consumption * 1.1)

        if status in ["Low", "Critical"]:
            if cv > 1.0:
                recommended_stock = int(avg_consumption * 1.3)
                return f"URGENT: Order {recommended_stock - current_quantity} units. High variability - maintain {recommended_stock} units."
            else:
                return f"Order {optimal_stock - current_quantity} units. Target: {optimal_stock} units."
        elif status == "Adequate":
            if cv > 1.0:
                return f"Stock adequate but monitor closely. Consider increasing to {int(avg_consumption * 1.2)} units before peak season."
            else:
                return f"Stock level appropriate. Maintain {optimal_stock} units as optimal."
        else:  # Excess
            if cv > 1.0:
                return f"Excess stock. Current: {current_quantity}, Optimal: {optimal_stock}. Reduce ordering but keep extra due to variability."
            else:
                return f"Excess stock. Current: {current_quantity}, Optimal: {optimal_stock}. Stop ordering until stock reduces."

    def _interpret_cv(self, cv_value: float) -> str:
        """Optimized CV interpretation"""
        if cv_value < 0.3:
            return "Highly predictable demand"
        elif cv_value < 0.7:
            return "Moderately predictable demand"
        elif cv_value < 1.2:
            return "Variable demand pattern"
        else:
            return "Highly unpredictable demand"

    def clear_cache(self):
        """Clear all caches to free memory"""
        self._prediction_cache.clear()
        self._search_cache.clear()
        # Also clear analyzer caches
        if hasattr(self.seasonal_analyzer, 'clear_cache'):
            self.seasonal_analyzer.clear_cache()