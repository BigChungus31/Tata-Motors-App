#New Analyzer.py

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Optional
from utils import setup_web_logging as setup_logging, handle_exceptions

# Module-level logger (shared across instances)
logger = setup_logging(__name__)

VENDOR_CATEGORIES = {
    'Tata AutoComp Systems': {
        'keywords': [
            # HVAC & Cooling
            'radiator', 'condenser', 'heat exchanger', 'cooling', 'hvac', 'air conditioning', 'thermal', 'compressor',
            'fan', 'blower', 'air curtain', 'louver', 'ventilation',
            # Climate Control
            'ac louver', 'air spring', 'passenger fan', 'driver fan', 'impeller',
            # Thermal Management
            'coolant', 'fan mounting bracket', 'cooling system'
        ],
        'contact': 'orders@tataautocomp.com',
        'phone': '+91-20-6604-7000'
    },
    'Motherson Sumi Systems': {
        'keywords': [
            # Wiring & Electronics
            'harness', 'wiring', 'cable', 'connector', 'electronic', 'module', 'ecu', 'junction box',
            'switch', 'relay', 'fuse', 'bulb', 'lamp', 'led', 'lighting',
            # Exterior Components
            'plastic', 'trim', 'exterior', 'bumper', 'panel', 'skirt', 'flap', 'cover',
            'canopy', 'door', 'handle', 'lock', 'hinge', 'emblem', 'graphic', 'sticker',
            # Specific Items
            'wiring harness', 'cable tie', 'micro relay', 'stop request', 'emergency help',
            'destination board', 'panic', 'can', 'pigtail', 'extension', 'hv cable',
            'fdas', 'traction motor', 'battery cable', 'charger cable', 'inverter',
            'vehicle electronic controller', 'battery cooling', 'clamp assy',
            # Body & Trim
            'side external paneling', 'upper side external', 'passenger door', 'driver door',
            'rear flap', 'front flap', 'end show cover', 'mounting reinforcement',
            'finish alu', 'aluminum profile', 'aluminium profile', 'beading', 'profile',
            'paneling', 'canopy', 'external latch', 'lock assy', 'lock cover'
        ],
        'contact': 'procurement@motherson.com',
        'phone': '+91-124-471-1000'
    },
    'Bharat Forge': {
        'keywords': [
            # Forged Components
            'forging', 'axle', 'crankshaft', 'connecting rod', 'gear', 'transmission', 'differential',
            'chassis', 'suspension', 'spring', 'shock', 'damper',
            # Suspension & Chassis
            'rear shock', 'air spring', 'torque rod', 'anti roll bar', 'arb', 'rubber bush',
            'bellow', 'leveling valve', 'brake actuator', 'slack adjuster',
            'rolling diaphragm', 'disc brake', 'brake shoe', 'brake lining', 'brake pad',
            # Fasteners & Hardware
            'hex bolt', 'u bolt', 'nyloc nut', 'clamping stud', 'rivet', 'bracket',
            'support', 'mounting', 'reinforcement'
        ],
        'contact': 'sales@bharatforge.com',
        'phone': '+91-20-6704-2000'
    },
    'ZF India': {
        'keywords': [
            # Steering Systems
            'steering', 'power steering', 'hydraulic', 'pump', 'valve', 'cylinder', 'piston',
            'steering oil', 'atf', 'dextron',
            # Brake Systems
            'brake', 'clutch', 'gearbox', 'modulator', 'abs', 'ebs', 'electronic braking',
            'brake pad', 'brake shoe', 'brake lining', 'disc brake', 'spring brake',
            # Transmission
            'gear range selector', 'transmission', 'auto slack adjuster', 'graduated hand control'
        ],
        'contact': 'india@zf.com',
        'phone': '+91-80-4077-4000'
    },
    'Bosch India': {
        'keywords': [
            # Engine Management
            'fuel injection', 'ignition', 'spark plug', 'sensor', 'ecu', 'alternator', 'starter',
            'oil level switch', 'oil pressure switch', 'reverse gear alarm',
            # Electronic Systems
            'electronic', 'controller', 'module', 'vts device', 'ais140', 'tds lite',
            'consep unit', 'flashing tool', 'parking sensor', 'reverse camera',
            'reverse park assistance', 'ipc', 'chemito',
            # Sensors & Switches
            'switch', 'sensor', 'solenoid valve', 'range selector', 'panic switch',
            'internal stage lamp', 'front door open', 'rear door open'
        ],
        'contact': 'orders@bosch.in',
        'phone': '+91-80-6749-2222'
    },
    'Apollo Tyres': {
        'keywords': [
            # Tyres & Wheels
            'tyre', 'tire', 'tube', 'rim', 'wheel', 'rubber', 'radial', 'valve', 'balance', 'tread',
            'tubeless', 'mtr', 'wheel arch'
        ],
        'contact': 'orders@apollotyres.com',
        'phone': '+91-44-2839-3000'
    },
    'Asahi India Glass': {
        'keywords': [
            # Glass Components
            'glass', 'windshield', 'windscreen', 'window', 'mirror', 'glazing',
            'door glass', 'quarter glass', 'tempered', 'laminated', 'pasted',
            'front windshield', 'rear windshield', 'sliding glass', 'tilting glass',
            'panoramic', 'ceramic', 'gray', 'tnt', 'nir', 'ir cut',
            # Mirror Systems
            'orvm', 'rvm', 'rear view mirror', 'side mirror', 'wide angle mirror',
            'main mirror', 'view mirror', 'mirror support', 'mirror arm'
        ],
        'contact': 'orders@aisglass.com',
        'phone': '+91-11-2682-9999'
    },
    'NTN Bearing India': {
        'keywords': [
            # Bearing Components
            'bearing', 'ball bearing', 'roller bearing', 'thrust bearing', 'needle bearing',
            'hub', 'wheel bearing', 'drive bearing', 'shaft bearing', 'journal bearing',
            'hub assy', 'rear hub', 'taper roller bearing', 'gasket rear hub'
        ],
        'contact': 'sales@ntnindia.com',
        'phone': '+91-44-2815-1234'
    },
    'Sundaram Clayton': {
        'keywords': [
            # Engine Components
            'piston', 'rings', 'liner', 'sleeve', 'cylinder', 'head', 'gasket', 'valve',
            'seat', 'guide', 'filter', 'air filter', 'oil filter', 'fuel filter',
            'breather', 'element', 'kit', 'regulator', 'dryer'
        ],
        'contact': 'orders@sundaramclayton.com',
        'phone': '+91-44-2220-1000'
    },
    'Tata Steel': {
        'keywords': [
            # Steel & Metal Components
            'steel', 'sheet', 'plate', 'bar', 'rod', 'tube', 'pipe', 'frame', 'body',
            'panel', 'metal', 'gi', 'galvanized', 'structural', 'fabricated',
            # Safety & Miscellaneous
            'safety hammer', 'fire extinguisher', 'grab rail', 'handrail', 'support',
            'reflector', 'reflective tape', 'yellow refl', 'white reflective', 'red reflective',
            'exit sticker', 'license plate', 'mud flap', 'rubber buffer',
            'clamp', 'clip', 'grommet', 'seal', 'sealant', 'gasket',
            # Fluids & Lubricants
            'oil', 'grease', 'coolant', 'lubricant', 'chassis grease', 'axle oil',
            'prop shaft grease', 'synthetic grease', 'multipurpose',
            # Hardware & Fasteners
            'screw', 'bolt', 'nut', 'washer', 'rivet', 'pin', 'stud',
            'fastener', 'hex', 'phillips', 'allen', 'flat head', 'cheese head',
            'button bolt', 'button nut', 'nyloc', 'din', 'iso',
            # Rubber & Sealing
            'rubber', 'o-ring', 'seal', 'gasket', 'buffer', 'grommet',
            'weather strip', 'door seal', 'window seal', 'bidding'
        ],
        'contact': 'automotive@tatasteel.com',
        'phone': '+91-33-6612-7000'
    }
}

class MaterialConsumptionAnalyzer:
    def __init__(self):
        self.processed_data = None
        self.material_descriptions = None
        self.unique_materials_count = 0
        # Cache for computed results
        self._seasonal_cache = None
        self._classification_cache = None

    @handle_exceptions
    def initialize_from_data(self, processed_data: pd.DataFrame, raw_data: pd.DataFrame) -> Dict:
        """Initialize analyzer with processed data from DataHandler"""
        try:
            # Only store essential data
            self.processed_data = processed_data
            
            # Create material descriptions lookup (only once)
            self.material_descriptions = processed_data.groupby('material_number')['material_description'].first().to_dict()
            self.unique_materials_count = processed_data['material_number'].nunique()
            
            return {
                "success": True,
                "status": "success",
                "unique_materials": self.unique_materials_count,
                "date_range": {
                    "start": processed_data['document_date'].min().strftime('%d-%m-%Y'),
                    "end": processed_data['document_date'].max().strftime('%d-%m-%Y')
                }
            }
        except Exception as e:
            logger.error(f"Error initializing analyzer: {str(e)}")
            return {"success": False, "error": f"Initialization failed: {str(e)}"}
    
    def _get_seasonal_patterns(self):
        """Compute seasonal patterns with caching"""
        if self._seasonal_cache is not None:
            return self._seasonal_cache
        
        try:
            # Single aggregation operation
            monthly_data = self.processed_data.groupby(['material_number', 'month'])['consumption_qty'].sum().unstack(fill_value=0)
            
            # Compute only essential statistics
            patterns = pd.DataFrame({
                'avg_monthly_consumption': monthly_data.mean(axis=1).round(0).astype(int),
                'total_annual_consumption': np.maximum(monthly_data.sum(axis=1), self.processed_data.groupby('material_number')['consumption_qty'].sum().reindex(monthly_data.index, fill_value=0)).round(0).astype(int),
                'peak_month': monthly_data.idxmax(axis=1),
                'coefficient_of_variation': np.where(
                    monthly_data.mean(axis=1) > 0,
                    (monthly_data.std(axis=1) / monthly_data.mean(axis=1)).round(2),
                    0
                )
            })
            
            # Cache the result
            self._seasonal_cache = patterns
            return patterns
            
        except Exception as e:
            logger.error(f"Error computing seasonal patterns: {str(e)}")
            return pd.DataFrame()

    @handle_exceptions
    def analyze_seasonal_patterns(self) -> Dict:
        """Analyze seasonal consumption patterns - lightweight version"""
        try:
            patterns = self._get_seasonal_patterns()
            
            if patterns.empty:
                return {"success": False, "error": "No patterns computed"}
            
            return {
                "success": True,
                "status": "success",
                "materials_analyzed": len(patterns),
                "avg_consumption_overall": int(patterns['avg_monthly_consumption'].mean()),
                "most_consumed_material": {
                    "material": patterns['avg_monthly_consumption'].idxmax(),
                    "avg_consumption": int(patterns['avg_monthly_consumption'].max())
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing seasonal patterns: {str(e)}")
            return {"success": False, "error": f"Seasonal analysis failed: {str(e)}"}

    @handle_exceptions
    def classify_materials_simplified(self) -> Dict:
        """Lightweight material classification"""
        if self._classification_cache is not None:
            return {"success": True, "status": "success", "classifications": self._classification_cache}
        
        try:
            patterns = self._get_seasonal_patterns()
            
            if patterns.empty:
                return {"success": False, "error": "No patterns available for classification"}
            
            # Simplified thresholds
            high_threshold = patterns['avg_monthly_consumption'].quantile(0.8)
            medium_threshold = patterns['avg_monthly_consumption'].quantile(0.5)
            
            classifications = {
                'HD_Critical': [],
                'HD_Variable': [],
                'MD_Regular': [],
                'MD_Seasonal': [],
                'LD_Stable': [],
                'LD_Sporadic': []
            }
            
            for material in patterns.index:
                avg_consumption = patterns.loc[material, 'avg_monthly_consumption']
                cv = patterns.loc[material, 'coefficient_of_variation']
                
                # Simplified classification logic
                if avg_consumption >= high_threshold:
                    category = 'HD_Variable' if cv >= 1.0 else 'HD_Critical'
                elif avg_consumption >= medium_threshold:
                    category = 'MD_Seasonal' if cv >= 1.0 else 'MD_Regular'
                else:
                    category = 'LD_Sporadic' if cv >= 1.0 else 'LD_Stable'
                
                classifications[category].append(material)
            
            # Cache the result
            self._classification_cache = {cat: len(materials) for cat, materials in classifications.items()}
            
            return {
                "success": True,
                "status": "success",
                "classifications": self._classification_cache
            }
        except Exception as e:
            logger.error(f"Error in classification: {str(e)}")
            return {"success": False, "error": f"Classification failed: {str(e)}"}
    
    @handle_exceptions
    def _classify_material_by_vendor(self, material_number: str, description: str) -> str:
        """Enhanced material classification with improved keyword matching and priority logic"""
        description_lower = str(description).lower()
        material_lower = str(material_number).lower()
        
        # Combined text for better matching
        combined_text = f"{description_lower} {material_lower}"
        
        vendor_scores = {}
        
        # Score each vendor based on keyword matches
        for vendor, info in VENDOR_CATEGORIES.items():
            score = 0
            matched_keywords = []
            
            for keyword in info['keywords']:
                keyword_lower = keyword.lower()
                
                # Exact phrase match gets highest score
                if keyword_lower in combined_text:
                    # Longer keywords get higher scores
                    keyword_score = len(keyword_lower.split())
                    
                    # Boost score for exact description matches
                    if keyword_lower in description_lower:
                        keyword_score *= 2
                    
                    score += keyword_score
                    matched_keywords.append(keyword)
            
            if score > 0:
                vendor_scores[vendor] = {
                    'score': score,
                    'keywords': matched_keywords
                }
        
        # If no matches found, apply fallback logic
        if not vendor_scores:
            return self._apply_fallback_classification(description_lower, material_lower)
        
        # Return vendor with highest score
        best_vendor = max(vendor_scores.items(), key=lambda x: x[1]['score'])
        return best_vendor[0]

    @handle_exceptions
    def _apply_fallback_classification(self, description_lower: str, material_lower: str) -> str:
        """Fallback classification for items that don't match primary keywords"""
        combined_text = f"{description_lower} {material_lower}"
        
        # Electronic/electrical components
        if any(term in combined_text for term in ['24v', '12v', 'volt', 'amp', 'electronic', 'electric']):
            return 'Motherson Sumi Systems'
        
        # Glass-related
        if any(term in combined_text for term in ['gls', 'transparent', 'clear']):
            return 'Asahi India Glass'
        
        # Mechanical parts
        if any(term in combined_text for term in ['assy', 'assembly', 'kit', 'component']):
            return 'Bharat Forge'
        
        # EV-specific components
        if any(term in combined_text for term in ['ev', 'hv', 'battery', 'motor', 'electric']):
            return 'Motherson Sumi Systems'
        
        # Bus-specific body parts
        if any(term in combined_text for term in ['bs1', 'bs2', 'bs3', 'bs4', 'bs6', 'gen5', 'gen 5']):
            return 'Tata Steel'
        
        # Default fallback
        return 'Tata Steel'

    @handle_exceptions
    def get_vendor_materials(self) -> Dict:
        """Get materials categorized by vendors with enhanced classification"""
        try:
            if not self.material_descriptions:
                return {"success": False, "error": "No material data available"}
            
            vendor_materials = {}
            classification_stats = {}
            
            # Initialize vendor categories
            for vendor in VENDOR_CATEGORIES.keys():
                vendor_materials[vendor] = {
                    'materials': [],
                    'contact': VENDOR_CATEGORIES[vendor]['contact'],
                    'phone': VENDOR_CATEGORIES[vendor]['phone']
                }
                classification_stats[vendor] = 0
            
            # Classify each material
            for material_number, description in self.material_descriptions.items():
                vendor = self._classify_material_by_vendor(material_number, description)
                
                vendor_materials[vendor]['materials'].append({
                    'material_number': material_number,
                    'description': description
                })
                
                classification_stats[vendor] += 1
            
            return {
                "success": True,
                "vendor_materials": vendor_materials,
                "total_vendors": len(vendor_materials),
                "classification_stats": classification_stats,
                "total_materials": len(self.material_descriptions)
            }
            
        except Exception as e:
            logger.error(f"Error getting vendor materials: {str(e)}")
            return {"success": False, "error": f"Vendor classification failed: {str(e)}"}
    
    def get_vendor_cart_items(self, cart_items: List[Dict]) -> Dict:
        """Organize cart items by vendor with enhanced classification and pricing"""
        try:
            vendor_cart = {}
            
            # Initialize vendor categories
            for vendor in VENDOR_CATEGORIES.keys():
                vendor_cart[vendor] = {
                    'items': [],
                    'total_amount': 0,
                    'contact': VENDOR_CATEGORIES[vendor]['contact'],
                    'phone': VENDOR_CATEGORIES[vendor]['phone']
                }
            
            # Process each cart item
            for item in cart_items:
                material_number = item.get('material_number')
                quantity = item.get('quantity', 0)

                if not material_number or quantity <= 0:
                    continue
                    
                if material_number in self.material_descriptions:
                    description = self.material_descriptions[material_number]
                    vendor = self._classify_material_by_vendor(material_number, description)
                    
                    # Calculate pricing
                    unit_price = self._get_unit_price(material_number)
                    total_price = quantity * unit_price
                    
                    vendor_cart[vendor]['items'].append({
                        'material_number': material_number,
                        'description': description,
                        'quantity': quantity,
                        'unit_price': round(unit_price, 2),
                        'total_price': round(total_price, 2)
                    })
                    
                    vendor_cart[vendor]['total_amount'] += total_price
            
            # Remove vendors with no items and round totals
            vendor_cart = {
                vendor: {
                    **data,
                    'total_amount': round(data['total_amount'], 2)
                }
                for vendor, data in vendor_cart.items() 
                if data['items']
            }
            
            return {
                "success": True,
                "vendor_cart": vendor_cart,
                "vendors_involved": len(vendor_cart)
            }
            
        except Exception as e:
            logger.error(f"Error organizing vendor cart: {str(e)}")
            return {"success": False, "error": f"Vendor cart organization failed: {str(e)}"}
    
    def _get_unit_price(self, material_number: str) -> float:
        """Helper method to get unit price for a material"""
        try:
            if hasattr(self, 'processed_data') and 'amount' in self.processed_data.columns:
                material_data = self.processed_data[
                    self.processed_data['material_number'] == material_number
                ]
                
                if not material_data.empty:
                    valid_transactions = material_data[
                        (material_data['consumption_qty'] > 0) & 
                        (material_data['amount'] > 0)
                    ]
                    
                    if not valid_transactions.empty:
                        unit_prices = valid_transactions['amount'] / valid_transactions['consumption_qty']
                        return float(unit_prices.max())
            
            return 0.0
            
        except (ValueError, TypeError, ZeroDivisionError):
            return 0.0
    
    @handle_exceptions
    def get_vendor_mappings(self, cart_dict: Dict) -> Dict:
        """Convert cart dictionary to vendor mappings with enhanced classification"""
        try:
            # Convert cart dictionary to list format
            cart_items = []
            for material_number, item_data in cart_dict.items():
                cart_items.append({
                    'material_number': material_number,
                    'quantity': item_data.get('quantity', 0)
                })
            
            # Get vendor cart organization
            result = self.get_vendor_cart_items(cart_items)
            
            if result.get('success', False):
                return {
                    'success': True,
                    'vendors': [
                        {
                            'name': vendor_name,
                            'materials': [
                                {
                                    'material_number': item['material_number'],
                                    'description': item['description'],
                                    'quantity': item['quantity'],
                                    'unit_price': item['unit_price'],
                                    'total': item['total_price']
                                }
                                for item in vendor_data.get('items', [])
                            ],
                            'total_value': vendor_data.get('total_amount', 0),
                            'contact': vendor_data.get('contact', ''),
                            'phone': vendor_data.get('phone', '')
                        }
                        for vendor_name, vendor_data in result['vendor_cart'].items()
                        if vendor_data.get('items')
                    ]
                }
            else:
                return result
                
        except Exception as e:
            logger.error(f"Error getting vendor mappings: {str(e)}")
            return {"success": False, "error": f"Vendor mapping failed: {str(e)}"}

    @handle_exceptions
    def predict_inventory(self, material_number: str, current_qty: float) -> Dict:
        
        logger.info(f"Predict inventory called for material: {material_number}, qty: {current_qty}")
        logger.info(f"Processed data shape: {self.processed_data.shape if self.processed_data is not None else 'None'}")
        logger.info(f"Material descriptions count: {len(self.material_descriptions) if self.material_descriptions else 0}")
                
        try:
            patterns = self._get_seasonal_patterns()
            
            if material_number not in patterns.index:
                return {"success": False, "error": "Material not found"}
            
            avg_monthly = patterns.loc[material_number, 'avg_monthly_consumption']
            
            # First determine the material type
            if avg_monthly == 0:
                material_type = "Non-Moving"
                coverage_months = "Unlimited"
                cv = patterns.loc[material_number, 'coefficient_of_variation']
                yearly_consumption = int(patterns.loc[material_number, 'total_annual_consumption'])
                recommendations = ["Order only when specifically needed", "Avoid routine ordering to prevent excess inventory"]
                qty_status = "No-Consumption"
            elif avg_monthly < 0.5:  # Less than 0.5 units per month
                material_type = "Infrequent"
                coverage_months = round(current_qty / avg_monthly, 1) if avg_monthly > 0 else "Unlimited"
                yearly_consumption = int(avg_monthly * 12)
                
                # Calculate quantity status for infrequent materials
                if current_qty > avg_monthly * 1.1:
                    qty_status = "Excess"
                    recommendations = ["Consider reducing inventory levels", f"Annual consumption is {yearly_consumption} units"]
                elif current_qty >= avg_monthly:
                    qty_status = "Adequate"
                    recommendations = ["Inventory levels are appropriate", f"Annual consumption is {yearly_consumption} units"]
                elif current_qty >= avg_monthly * 0.5:
                    qty_status = "Low"
                    recommendations = ["Consider reordering soon", f"Annual consumption is {yearly_consumption} units"]
                else:
                    qty_status = "Critical"
                    recommendations = ["Immediate reorder required", f"Annual consumption is {yearly_consumption} units"]
            elif avg_monthly < 2:  # Less than 2 units per month
                material_type = "Slow-Moving"
                coverage_months = round(current_qty / avg_monthly, 1)
                yearly_consumption = int(avg_monthly * 12)
                
                # Calculate quantity status for slow-moving materials
                if current_qty > avg_monthly * 1.1:
                    qty_status = "Excess"
                    recommendations = ["Consider reducing inventory levels", f"Annual consumption is {yearly_consumption} units"]
                elif current_qty >= avg_monthly:
                    qty_status = "Adequate"
                    recommendations = ["Inventory levels are appropriate", f"Annual consumption is {yearly_consumption} units"]
                elif current_qty >= avg_monthly * 0.5:
                    qty_status = "Low"
                    recommendations = ["Consider reordering soon", f"Annual consumption is {yearly_consumption} units"]
                else:
                    qty_status = "Critical"
                    recommendations = ["Immediate reorder required", f"Annual consumption is {yearly_consumption} units"]
            else:
                # Regular materials
                material_type = "Regular"
                coverage_months = round(current_qty / avg_monthly, 1)
                yearly_consumption = int(avg_monthly * 12)
                
                if current_qty > avg_monthly * 1.1:
                    qty_status = "Excess"
                    recommendations = ["Consider reducing inventory levels"]
                elif current_qty >= avg_monthly:
                    qty_status = "Adequate"
                    recommendations = ["Inventory levels are appropriate"]
                elif current_qty >= avg_monthly * 0.5:
                    qty_status = "Low"
                    recommendations = ["Consider reordering soon"]
                else:
                    qty_status = "Critical"
                    recommendations = ["Immediate reorder required"]

            # Set status to show both quantity status and material type
            status = f"{qty_status} - {material_type}"
            
            # Calculate order quantity and get CV
            if avg_monthly == 0:
                order_qty = 0
                cv = 0
            else:
                cv = patterns.loc[material_number, 'coefficient_of_variation']
                if avg_monthly < 2:  # Slow-moving or infrequent - no order suggestions
                    order_qty = 0
                else:
                    target_qty = avg_monthly * 1.1
                    order_qty = max(0, int(target_qty - current_qty))
            
            unit_price = 0
            if hasattr(self, 'processed_data') and 'amount' in self.processed_data.columns:
                try:
                    material_data = self.processed_data[self.processed_data['material_number'] == material_number]
                    if not material_data.empty:
                        # Calculate unit price using amount/consumption_qty for each transaction
                        valid_transactions = material_data[
                            (material_data['consumption_qty'] > 0) & 
                            (material_data['amount'] > 0) & 
                            (material_data['consumption_qty'].notna()) & 
                            (material_data['amount'].notna())
                        ]
                        if not valid_transactions.empty:
                            # Calculate unit price directly from amount and consumption_qty
                            unit_prices = valid_transactions['amount'] / valid_transactions['consumption_qty']
                            unit_price = float(unit_prices.max())
                            
                            unit_price = float(unit_prices.max())
                except (ValueError, TypeError, ZeroDivisionError):
                    unit_price = 0

            return {
                "success": True,
                "prediction": {
                    "material_number": material_number,
                    "current_quantity": current_qty,
                    "avg_monthly_consumption": int(avg_monthly),
                    "order_quantity": order_qty,
                    "unit_price": round(unit_price, 2),
                    "order_amount": round(order_qty * unit_price, 2),
                    "coverage_months": coverage_months,
                    "status": status,
                    "material_type": material_type,
                    "quantity_status": qty_status,
                    "recommendations": recommendations,
                    "demand_variability": f"CV: {cv:.2f}" if cv >= 0 else "CV: N/A",
                    "yearly_consumption": yearly_consumption,
                    "add_to_cart_eligible": qty_status in ['Critical', 'Low'] and order_qty > 0,
                    "vendor": self._classify_material_by_vendor(material_number, self.material_descriptions.get(material_number, ''))
                }
            }
        except Exception as e:
            logger.error(f"Error predicting inventory: {str(e)}")
            return {"success": False, "error": f"Prediction failed: {str(e)}"}

    @handle_exceptions
    def generate_summary_report(self) -> Dict:
        """Generate lightweight summary report instead of comprehensive one"""
        try:
            patterns = self._get_seasonal_patterns()
            classifications = self.classify_materials_simplified()
            
            if patterns.empty:
                return {"success": False, "error": "No data available for report"}
            
            # Basic summary only
            report = {
                'success': True,
                'summary': {
                    'total_materials': self.unique_materials_count,
                    'date_range': f"{self.processed_data['document_date'].min().strftime('%d-%m-%Y')} to {self.processed_data['document_date'].max().strftime('%d-%m-%Y')}",
                    'high_velocity_items': len([m for m in patterns.index if patterns.loc[m, 'avg_monthly_consumption'] >= patterns['avg_monthly_consumption'].quantile(0.9)]),
                    'avg_monthly_consumption': int(patterns['avg_monthly_consumption'].mean())
                },
                'classifications': classifications.get('classifications', {}),
                'top_consumers': patterns.nlargest(10, 'avg_monthly_consumption')[['avg_monthly_consumption', 'total_annual_consumption']].to_dict('index')
            }
            
            return report
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return {"success": False, "error": f"Report generation failed: {str(e)}"}

    @handle_exceptions
    def search_materials(self, query: str) -> Dict:
        """Optimized material search"""
        try:
            if not query or len(query) < 2 or not self.material_descriptions:
                return {"success": True, "materials": []}
            
            query_lower = query.lower()
            matches = []
            
            # Limit search to first 50 matches for performance
            for material_number, description in list(self.material_descriptions.items())[:1000]:  # Limit search scope
                if (query_lower in str(material_number).lower() or 
                    query_lower in str(description).lower()):
                    matches.append({
                        'material_number': material_number,
                        'description': description
                    })
                    
                    if len(matches) >= 20:  # Early exit
                        break

            return {"success": True, "materials": matches, "total_results": len(matches)}
        except Exception as e:
            logger.error(f"Error searching materials: {str(e)}")
            return {"success": False, "error": f"Search failed: {str(e)}"}

    def _get_non_moving_materials(self, raw_data: pd.DataFrame) -> List[Dict]:
        """Compute non-moving materials on-demand (only for full report)"""
        try:
            all_materials = set(raw_data['material_number'].dropna().astype(str).unique())
            active_materials = set(self.processed_data['material_number'].unique())
            non_moving = all_materials - active_materials
            
            non_moving_list = []
            for material in non_moving:
                if material != 'nan':
                    desc_data = raw_data[raw_data['material_number'] == material]['material_description']
                    desc = desc_data.iloc[0] if not desc_data.empty else 'N/A'
                    non_moving_list.append({
                        'material_number': material,
                        'description': desc,
                        'avg_consumption': 0,
                        'total_consumption': 0,
                        'peak_month': 'N/A',
                        'coefficient_of_variation': 0,
                        'coverage_needed': 0,
                        'status': 'Non-Moving'
                    })
            
            return non_moving_list
        except Exception as e:
            logger.error(f"Error identifying non-moving materials: {str(e)}")
            return []

    @handle_exceptions
    def generate_full_report(self, raw_data: pd.DataFrame) -> Dict:
        """Generate comprehensive report with non-moving materials"""
        try:
            logger.info("Generating full consumption report")
            
            patterns = self._get_seasonal_patterns()
            classifications_result = self.classify_materials_simplified()
            
            if patterns.empty:
                return {"success": False, "error": "No data available for report"}
            
            # Get non-moving materials (only for full report)
            non_moving_materials = self._get_non_moving_materials(raw_data)
            
            # Create detailed classifications with material info
            detailed_classifications = {
                'hd_critical': [],
                'hd_variable': [], 
                'md_regular': [],
                'md_seasonal': [],
                'ld_stable': [],
                'ld_sporadic': []
            }

            # Get classification data for detailed breakdown
            if classifications_result.get('success'):
                # We need to rebuild classifications for detailed info
                high_threshold = patterns['avg_monthly_consumption'].quantile(0.8)
                medium_threshold = patterns['avg_monthly_consumption'].quantile(0.5)
                
                classification_mapping = {
                    'HD_Critical': 'hd_critical',
                    'HD_Variable': 'hd_variable',
                    'MD_Regular': 'md_regular',
                    'MD_Seasonal': 'md_seasonal',
                    'LD_Stable': 'ld_stable',
                    'LD_Sporadic': 'ld_sporadic'
                }
                
                month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April',
                              5: 'May', 6: 'June', 7: 'July', 8: 'August',
                              9: 'September', 10: 'October', 11: 'November', 12: 'December'}
                
                for material in patterns.index:
                    stats = patterns.loc[material]
                    avg_consumption = stats['avg_monthly_consumption']
                    cv = stats['coefficient_of_variation']
                    
                    # Determine category
                    if avg_consumption >= high_threshold:
                        category = 'HD_Variable' if cv >= 1.0 else 'HD_Critical'
                    elif avg_consumption >= medium_threshold:
                        category = 'MD_Seasonal' if cv >= 1.0 else 'MD_Regular'
                    else:
                        category = 'LD_Sporadic' if cv >= 1.0 else 'LD_Stable'
                    
                    target_key = classification_mapping[category]
                    
                    detailed_classifications[target_key].append({
                        'material_number': material,
                        'description': self.material_descriptions.get(material, 'N/A'),
                        'avg_consumption': int(stats['avg_monthly_consumption']),
                        'total_consumption': int(stats['total_annual_consumption']),
                        'peak_month': month_names.get(stats['peak_month'], 'N/A'),
                        'coefficient_of_variation': round(stats['coefficient_of_variation'], 2),
                        'coverage_needed': int(stats['avg_monthly_consumption'] * 1.1) if stats['avg_monthly_consumption'] > 0 else 0
                    })
            
            # Create seasonal analysis data
            seasonal_analysis = []
            month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April',
                          5: 'May', 6: 'June', 7: 'July', 8: 'August',
                          9: 'September', 10: 'October', 11: 'November', 12: 'December'}
            
            # Simple seasonal analysis based on peak months
            for month_num, month_name in month_names.items():
                peak_materials = len([mat for mat in patterns.index 
                                    if patterns.loc[mat, 'peak_month'] == month_num])
                seasonal_analysis.append({
                    'month': month_name,
                    'peak_materials': peak_materials,
                    'total_consumption': 0  # Could compute if needed
                })
            
            report = {
                'success': True,
                'summary': {
                    'total_materials': self.unique_materials_count,
                    'date_range': f"{self.processed_data['document_date'].min().strftime('%d-%m-%Y')} to {self.processed_data['document_date'].max().strftime('%d-%m-%Y')}",
                    'high_velocity_items': len([m for m in patterns.index if patterns.loc[m, 'avg_monthly_consumption'] >= patterns['avg_monthly_consumption'].quantile(0.9)]),
                    'cost_optimization_potential': len(detailed_classifications['ld_stable']) + len(detailed_classifications['ld_sporadic']),
                    'inventory_efficiency_score': round((len(detailed_classifications['hd_critical']) + len(detailed_classifications['md_regular'])) / max(self.unique_materials_count, 1) * 100, 1),
                    'non_moving_items': len(non_moving_materials)
                },
                
                # Add detailed material breakdowns including non-moving materials
                'material_details': {
                    'high_demand_critical': detailed_classifications['hd_critical'],
                    'high_demand_variable': detailed_classifications['hd_variable'],
                    'medium_demand_regular': detailed_classifications['md_regular'],
                    'medium_demand_seasonal': detailed_classifications['md_seasonal'],
                    'low_demand_stable': detailed_classifications['ld_stable'],
                    'low_demand_sporadic': detailed_classifications['ld_sporadic'],
                    'non_consuming_materials': non_moving_materials  # This is what you need!
                },
                
                'classifications': detailed_classifications,
                'seasonal_analysis': seasonal_analysis,
                'simplified_classifications': classifications_result.get('classifications', {})
            }
            
            logger.info(f"Full report generated successfully with {len(non_moving_materials)} non-moving materials")
            return report
            
        except Exception as e:
            logger.error(f"Error generating full report: {str(e)}")
            return {"success": False, "error": f"Report generation failed: {str(e)}"}

    def clear_cache(self):
        """Clear cached results to free memory"""
        self._seasonal_cache = None
        self._classification_cache = None