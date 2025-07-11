import sys
import os
print(f"Python path: {sys.path}")
print(f"Current working directory: {os.getcwd()}")
print(f"File location: {__file__}")
from flask import Blueprint, render_template, request, jsonify, session, send_from_directory, current_app
from werkzeug.utils import secure_filename
import uuid
import tempfile
import time
backend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'backend')
print(f"Backend path: {backend_path}")
print(f"Backend path exists: {os.path.exists(backend_path)}")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)
try:
    from analyzer import MaterialConsumptionAnalyzer
    print("✓ Successfully imported MaterialConsumptionAnalyzer")
except ImportError as e:
    print(f"✗ Failed to import MaterialConsumptionAnalyzer: {e}")

try:
    from data_handler import DataLoader, DataValidator
    print("✓ Successfully imported DataLoader and DataValidator")
except ImportError as e:
    print(f"✗ Failed to import DataLoader/DataValidator: {e}")

try:
    import pandas as pd
    print("✓ Successfully imported pandas")
except ImportError as e:
    print(f"✗ Failed to import pandas: {e}")

main = Blueprint('main', __name__)
print("✓ Blueprint created successfully")

analyzer_cache = {}

def cleanup_old_sessions():
    current_time = time.time()
    expired_sessions = []
    
    for session_id, data in analyzer_cache.items():
        if current_time - data.get('timestamp', 0) > 3600:
            expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        try:
            if 'temp_file' in analyzer_cache[session_id]:
                temp_file = analyzer_cache[session_id]['temp_file']
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            del analyzer_cache[session_id]
        except:
            pass

def get_analyzer_from_session():
    try:
        session_id = session.get('analyzer_session_id')
        if not session_id or not session.get('file_loaded'):
            return None, "Session expired. Please re-upload your file."
        
        cleanup_old_sessions()
        
        if session_id not in analyzer_cache:
            return None, "Session data not found. Please re-upload your file."
        
        cached_data = analyzer_cache[session_id]
        
        if time.time() - cached_data.get('timestamp', 0) > 3600:
            return None, "Session expired. Please re-upload your file."
        
        return cached_data['analyzer'], None
        
    except Exception as e:
        return None, f"Error loading analyzer: {str(e)}"

def store_analyzer_in_cache(analyzer, raw_data=None):
    try:
        session_id = str(uuid.uuid4())
        
        analyzer_cache[session_id] = {
            'analyzer': analyzer,
            'raw_data': raw_data,
            'timestamp': time.time()
        }
        
        session['analyzer_session_id'] = session_id
        session['file_loaded'] = True
        
        return True
    except Exception as e:
        print(f"Error storing analyzer in cache: {str(e)}")
        return False

def get_raw_data_from_cache():
    try:
        session_id = session.get('analyzer_session_id')
        if session_id and session_id in analyzer_cache:
            return analyzer_cache[session_id].get('raw_data')
        return None
    except:
        return None

def load_data_from_supabase(depot_name=None):
    """Helper function to load and process data from Supabase"""
    try:
        data_loader = DataLoader()
        data_validator = DataValidator()
        
        # Load data from Supabase
        raw_data = data_loader.load_supabase_data(depot_name or 'Hassanpur')
        if raw_data is None or raw_data.empty:
            return None, None, "No data found in Supabase database"
        
        # Filter by depot if specified
        if depot_name:
            print(f"Filtering data for depot: {depot_name}")
            depot_data = raw_data[raw_data['Name 1'] == depot_name]
            if depot_data.empty:
                available_depots = raw_data['Name 1'].unique().tolist()
                return None, None, f"No data found for depot: {depot_name}. Available: {available_depots}"
            raw_data = depot_data
            print(f"Filtered data shape: {raw_data.shape}")
            
        # Process data
        detected_columns = data_loader.detect_columns_from_df(raw_data)
        processed_data = data_validator.process_raw_data(raw_data, detected_columns)
        
        # Create analyzer
        analyzer = MaterialConsumptionAnalyzer()
        result = analyzer.initialize_from_data(processed_data, raw_data)
        
        if result.get('success', False):
            return analyzer, raw_data, None
        else:
            return None, None, result.get('error', 'Processing failed')
            
    except Exception as e:
        return None, None, f'Data loading failed: {str(e)}'

@main.route('/')
def index():
    # Clear old session data
    session_id = session.get('analyzer_session_id')
    if session_id and session_id in analyzer_cache:
        try:
            if 'temp_file' in analyzer_cache[session_id]:
                temp_file = analyzer_cache[session_id]['temp_file']
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            del analyzer_cache[session_id]
        except:
            pass
    
    session.clear()
    return render_template('index.html')

@main.route('/upload', methods=['POST'])
def upload_file():
    """Load data from Supabase (maintains compatibility with frontend)"""
    try:
        analyzer, raw_data, error = load_data_from_supabase()
        if error:
            return jsonify({'success': False, 'message': error}), 500
        
        if store_analyzer_in_cache(analyzer, raw_data):
            session['filename'] = 'supabase_data.xlsx'
            session['file_loaded'] = True
            return jsonify({
                'success': True, 
                'message': 'Data loaded successfully from database'
            })
        else:
            return jsonify({'success': False, 'message': 'Failed to store data in cache'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Data loading failed: {str(e)}'}), 500

@main.route('/analyze', methods=['POST'])
def analyze_data():
    """Alternative endpoint for direct data analysis"""
    try:
        analyzer, raw_data, error = load_data_from_supabase()
        if error:
            return jsonify({'success': False, 'message': error}), 500
        
        if store_analyzer_in_cache(analyzer, raw_data):
            session['file_loaded'] = True
            return jsonify({
                'success': True, 
                'message': 'Data loaded successfully from Supabase'
            })
        else:
            return jsonify({'success': False, 'message': 'Failed to store data in cache'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Data loading failed: {str(e)}'}), 500

@main.route('/dashboard')
def dashboard():
    try:
        # Check if we have a valid session and depot
        if not session.get('file_loaded') or not session.get('depot'):
            return render_template('index.html', error='Please select a depot first')
        
        # Verify we have data in cache
        analyzer, error = get_analyzer_from_session()
        if not analyzer:
            return render_template('index.html', error='Session expired. Please select a depot again.')
        
        # Get raw data for dashboard
        raw_data = get_raw_data_from_cache()
        if raw_data is None or raw_data.empty:
            return render_template('index.html', error='No data available for dashboard.')
        
        print(f"Dashboard - Raw data shape: {raw_data.shape}")
        print(f"Dashboard - Depot: {session.get('depot')}")
        
        return render_template('analysis.html')
        
    except Exception as e:
        print(f"Dashboard error: {str(e)}")
        return render_template('index.html', error=str(e))

@main.route('/search', methods=['POST'])
def search_materials():
    print("=== SEARCH ROUTE CALLED ===")
    try:
        # Get analyzer from session
        analyzer, error = get_analyzer_from_session()
        if not analyzer:
            print("No analyzer found")
            return jsonify({'success': False, 'error': 'No data loaded. Please select a depot first.'})
        
        print("Analyzer found successfully")
        
        # Get search parameters
        data = request.get_json()
        if not data:
            print("No search data received")
            return jsonify({'success': False, 'error': 'No search query provided'})
            
        query = data.get('query', '').strip()
        print(f"Search query: '{query}'")
        
        # Get raw data from cache
        raw_data = get_raw_data_from_cache()
        if raw_data is None or raw_data.empty:
            print("No raw data in cache")
            return jsonify({'success': False, 'error': 'No data available'})
        
        print(f"Raw data shape: {raw_data.shape}")
        print(f"Raw data columns: {raw_data.columns.tolist()}")
        
        # Use the already filtered data (no need to filter again by depot)
        filtered_data = raw_data.copy()
        
        if query:
            print("Applying search filter...")
            search_mask = (
                filtered_data['Material'].astype(str).str.contains(query, case=False, na=False) |
                filtered_data['Material Description'].astype(str).str.contains(query, case=False, na=False)
            )
            filtered_data = filtered_data[search_mask]
            print(f"Filtered data shape after search: {filtered_data.shape}")
        else:
            print("No query provided, returning all data")
        
        # Remove duplicates by keeping first occurrence of each material
        filtered_data = filtered_data.drop_duplicates(subset=['Material'], keep='first')

        # Limit results and convert to required format
        filtered_data = filtered_data.head(50)
        print(f"Final data shape (unique materials): {filtered_data.shape}")

        materials = []

        for idx, row in filtered_data.iterrows():
            try:
                doc_date = pd.to_datetime(row['Document Date'])
                quantity = float(row['Quantity']) if pd.notna(row['Quantity']) else 0.0
                amount = float(row['Amount']) if pd.notna(row['Amount']) else 0.0
                
                material = {
                    'material_number': str(row['Material']),
                    'description': str(row['Material Description']),
                    'quantity': quantity,
                    'amount': amount,
                    'unit_price': analyzer._get_unit_price(str(row['Material'])),
                    'date': doc_date.strftime('%Y-%m-%d'),
                    'month': doc_date.strftime('%B'),
                    'year': int(doc_date.year)
                }
                materials.append(material)
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
        
        print(f"Total materials processed: {len(materials)}")
        
        result = {
            'success': True,
            'materials': materials,
            'total_results': len(materials),
            'query': query
        }
        
        print(f"Returning result with {len(materials)} materials")
        return jsonify(result)
        
    except Exception as e:
        print(f"Exception in search route: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@main.route('/predict', methods=['POST'])
def predict_inventory():
    try:
        # Get analyzer from session
        analyzer, error = get_analyzer_from_session()
        if not analyzer:
            return jsonify({'success': False, 'error': 'No data loaded. Please select a depot first.'})
        
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No prediction parameters provided'})
            
        material_number = data.get('material_number')
        current_qty = data.get('current_qty')
        
        if not material_number:
            return jsonify({'success': False, 'error': 'Material number is required'})
            
        result = analyzer.predict_inventory(material_number, current_qty)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@main.route('/report')
def comprehensive_report():
    try:
        analyzer, error = get_analyzer_from_session()
        if not analyzer:
            # Try to load data from Supabase
            analyzer, raw_data, load_error = load_data_from_supabase()
            if load_error:
                return render_template('results.html', error=load_error)
            
            store_analyzer_in_cache(analyzer, raw_data)
            session['file_loaded'] = True
        else:
            raw_data = get_raw_data_from_cache()
        
        if raw_data is not None:
            result = analyzer.generate_full_report(raw_data)
        else:
            result = analyzer.generate_summary_report()
        
        if result.get('success', False):
            return render_template('results.html', report=result)
        else:
            return render_template('results.html', error=result.get('error', 'Report generation failed'))
            
    except Exception as e:
        return render_template('results.html', error=str(e))

@main.route('/cart')
def cart():
    try:
        if not session.get('file_loaded'):
            return render_template('cart.html', error='Please upload a file first')
        
        return render_template('cart.html')
        
    except Exception as e:
        return render_template('cart.html', error=str(e))

@main.route('/add_to_cart', methods=['POST'])
def add_to_cart():
    try:
        data = request.get_json()
        material_number = data.get('material_number')
        quantity = float(data.get('quantity', 0))
        unit_price = float(data.get('unit_price', 0))
        description = data.get('description', '')
        
        if 'cart' not in session:
            session['cart'] = {}
        
        cart = session.get('cart', {})
        if material_number in cart:
            cart[material_number]['quantity'] += quantity
            cart[material_number]['total'] = cart[material_number]['quantity'] * cart[material_number]['unit_price']
        else:
            cart[material_number] = {
                'quantity': quantity,
                'unit_price': unit_price,
                'description': description,
                'total': quantity * unit_price
            }
        session['cart'] = cart
        session.modified = True
        return jsonify({'success': True, 'message': 'Material added to cart'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@main.route('/get_cart')
def get_cart():
    try:
        cart = session.get('cart', {})
        return jsonify({'success': True, 'cart': cart})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@main.route('/update_cart', methods=['POST'])
def update_cart():
    try:
        data = request.get_json()
        material_number = data.get('material_number')
        quantity = int(data.get('quantity', 0))
        
        if 'cart' not in session:
            return jsonify({'success': False, 'error': 'Cart not found'})
        
        cart = session.get('cart', {})
        
        if material_number in cart:
            if quantity > 0:
                cart[material_number]['quantity'] = quantity
                cart[material_number]['total'] = quantity * cart[material_number]['unit_price']
            else:
                del cart[material_number]
            
            session['cart'] = cart
            session.modified = True
            return jsonify({'success': True, 'message': 'Cart updated'})
        else:
            return jsonify({'success': False, 'error': 'Material not found in cart'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@main.route('/summary_report')
def summary_report():
    try:
        analyzer, error = get_analyzer_from_session()
        if not analyzer:
            # Try to load data from Supabase
            analyzer, raw_data, load_error = load_data_from_supabase()
            if load_error:
                return jsonify({'success': False, 'error': load_error})
            
            store_analyzer_in_cache(analyzer, raw_data)
            session['file_loaded'] = True
        
        result = analyzer.generate_summary_report()
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@main.route('/seasonal_analysis', methods=['GET'])
def seasonal_analysis():
    try:
        analyzer, error = get_analyzer_from_session()
        if not analyzer:
            # Try to load data from Supabase
            analyzer, raw_data, load_error = load_data_from_supabase()
            if load_error:
                return jsonify({'success': False, 'error': load_error})
            
            store_analyzer_in_cache(analyzer, raw_data)
            session['file_loaded'] = True
        
        result = analyzer.analyze_seasonal_patterns()
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@main.route('/classify_materials', methods=['GET'])
def classify_materials():
    try:
        analyzer, error = get_analyzer_from_session()
        if not analyzer:
            # Try to load data from Supabase
            analyzer, raw_data, load_error = load_data_from_supabase()
            if load_error:
                return jsonify({'success': False, 'error': load_error})
            
            store_analyzer_in_cache(analyzer, raw_data)
            session['file_loaded'] = True
        
        result = analyzer.classify_materials_simplified()
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@main.route('/clear_cache', methods=['POST'])
def clear_cache():
    try:
        analyzer, error = get_analyzer_from_session()
        if analyzer:
            analyzer.clear_cache()
            return jsonify({'success': True, 'message': 'Cache cleared'})
        else:
            return jsonify({'success': False, 'error': 'No analyzer found'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@main.route('/export', methods=['POST'])
def export_excel():
    try:
        analyzer, error = get_analyzer_from_session()
        if not analyzer:
            return jsonify({'success': False, 'error': 'No analyzer found'})
            
        data = request.get_json()
        filename = data.get('filename', 'material_analysis.xlsx')
        
        return jsonify({
            'success': False, 
            'error': 'Export functionality not yet implemented in analyzer'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@main.route('/download/<filename>')
def download_file(filename):
    try:
        export_folder = current_app.config.get('EXPORT_FOLDER', 'exports')
        return send_from_directory(export_folder, filename, as_attachment=True)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@main.route('/clear_session', methods=['POST'])
def clear_session():
    try:
        session_id = session.get('analyzer_session_id')
        if session_id and session_id in analyzer_cache:
            try:
                if 'temp_file' in analyzer_cache[session_id]:
                    temp_file = analyzer_cache[session_id]['temp_file']
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                del analyzer_cache[session_id]
            except:
                pass
        
        session.clear()
        return jsonify({'success': True, 'message': 'Session cleared'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@main.route('/set_depot', methods=['POST'])
def set_depot():
    """Set depot and load data from Supabase"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No depot data provided'})
            
        depot_key = data.get('depot')
        if not depot_key:
            return jsonify({'success': False, 'error': 'Depot name is required'})
        
        # Map frontend depot keys to actual database depot names
        depot_mapping = {
            'hassanpur': 'Hassanpur Depot',
            'Hassanpur Depot': 'Hassanpur Depot'
        }
        
        actual_depot_name = depot_mapping.get(depot_key, depot_key)
        print(f"Selected depot key: '{depot_key}' -> Actual depot: '{actual_depot_name}'")
        
        # Load data from Supabase with depot filtering
        analyzer, raw_data, error = load_data_from_supabase(actual_depot_name)
        if error:
            return jsonify({'success': False, 'error': f'Database error: {error}'})
        
        if raw_data is None or raw_data.empty:
            return jsonify({'success': False, 'error': 'No data found in database'})
        
        print(f"Loaded {len(raw_data)} records for depot: {actual_depot_name}")
        
        # Store filtered data
        if store_analyzer_in_cache(analyzer, raw_data):
            session['depot'] = actual_depot_name
            session['file_loaded'] = True
            return jsonify({'success': True, 'message': f'Depot {actual_depot_name} data loaded successfully'})
        else:
            return jsonify({'success': False, 'error': 'Failed to store data in cache'})
            
    except Exception as e:
        print(f"Exception in set_depot: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@main.route('/debug-supabase', methods=['GET'])
def debug_supabase():
    """Debug route to test Supabase connection"""
    try:
        analyzer, raw_data, error = load_data_from_supabase()
        if error:
            return {
                'success': False,
                'error': error
            }
        
        data_loader = DataLoader()
        detected_cols = data_loader.detect_columns_from_df(raw_data)
        
        return {
            'success': True,
            'data_shape': raw_data.shape,
            'columns': list(raw_data.columns),
            'detected_columns': detected_cols,
            'sample_data': raw_data.head().to_dict('records')
        }
        
    except Exception as e:
        print(f"Error in debug route: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }

@main.route('/auto_load', methods=['GET'])
def auto_load():
    """Auto-load data for frontend"""
    try:
        if not session.get('file_loaded'):
            analyzer, raw_data, error = load_data_from_supabase()
            if error:
                return jsonify({'success': False, 'error': error})
            
            if store_analyzer_in_cache(analyzer, raw_data):
                session['file_loaded'] = True
                session['filename'] = 'supabase_data.xlsx'
                return jsonify({'success': True, 'message': 'Data loaded successfully'})
            else:
                return jsonify({'success': False, 'error': 'Failed to store data'})
        else:
            return jsonify({'success': True, 'message': 'Data already loaded'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@main.route('/vendors')
def vendors():
    try:
        if not session.get('file_loaded'):
            return render_template('index.html', error='Please select a depot first')
        
        cart = session.get('cart', {})
        if not cart:
            return render_template('cart.html', error='Cart is empty')
        
        return render_template('vendor.html')
        
    except Exception as e:
        return render_template('index.html', error=str(e))

@main.route('/get_vendors')
def get_vendors():
    try:
        analyzer, error = get_analyzer_from_session()
        if not analyzer:
            return jsonify({'success': False, 'error': 'No data loaded. Please select a depot first.'})
        
        cart = session.get('cart', {})
        if not cart:
            return jsonify({'success': False, 'error': 'Cart is empty'})
        
        # Check if vendor mapping functionality exists
        if not hasattr(analyzer, 'get_vendor_mappings'):
            return jsonify({
                'success': False,
                'error': 'Vendor mapping functionality not available in analyzer'
            })

        # Get vendor mappings from analyzer
        vendor_mappings = analyzer.get_vendor_mappings(cart)

        if vendor_mappings and vendor_mappings.get('success', False):
            vendors = vendor_mappings.get('vendors', [])
            
            # Handle both list and dict formats
            if isinstance(vendors, list):
                # Ensure each vendor item has materials array
                for vendor_item in vendors:
                    if isinstance(vendor_item, dict) and 'materials' not in vendor_item:
                        vendor_item['materials'] = []
                    elif isinstance(vendor_item, dict) and vendor_item.get('materials') is None:
                        vendor_item['materials'] = []
            elif isinstance(vendors, dict):
                # Ensure each vendor has a materials array
                for vendor_name, vendor_data in vendors.items():
                    if 'materials' not in vendor_data:
                        vendor_data['materials'] = []
                    elif vendor_data['materials'] is None:
                        vendor_data['materials'] = []
            
            return jsonify({
                'success': True,
                'vendors': vendors
            })
        else:
            return jsonify({'success': False, 'error': vendor_mappings.get('error', 'Failed to get vendor mappings')})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@main.route('/confirm_order', methods=['POST'])
def confirm_order():
    try:
        data = request.get_json()
        vendor = data.get('vendor')
        materials = data.get('materials', [])
        
        if not vendor or not materials:
            return jsonify({'success': False, 'error': 'Invalid order data'})
        
        # Remove confirmed items from cart
        cart = session.get('cart', {})
        for material in materials:
            material_number = material.get('material_number')
            if material_number in cart:
                del cart[material_number]
        
        session['cart'] = cart
        session.modified = True
        
        return jsonify({
            'success': True,
            'message': f'Order confirmed with {vendor}',
            'order_id': str(uuid.uuid4())[:8]
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@main.route('/debug-data', methods=['GET'])
def debug_data():
    try:
        analyzer, raw_data, error = load_data_from_supabase()
        if error:
            return jsonify({'error': error})
        
        return jsonify({
            'columns': list(raw_data.columns),
            'sample_row': raw_data.head(1).to_dict('records')[0] if not raw_data.empty else {},
            'depot_values': raw_data['Name 1'].unique().tolist() if 'Name 1' in raw_data.columns else 'No "Name 1" column'
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@main.route('/debug-depot', methods=['GET'])
def debug_depot():
    try:
        analyzer, raw_data, error = load_data_from_supabase()
        if error:
            return jsonify({'error': error})
        
        unique_depots = raw_data['Name 1'].unique().tolist()
        return jsonify({
            'success': True,
            'available_depots': unique_depots,
            'sample_depot_value': unique_depots[0] if unique_depots else 'No depots found'
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@main.route('/test-supabase', methods=['GET'])
def test_supabase():
    try:
        data_loader = DataLoader()
        raw_data = data_loader.load_supabase_data('Hassanpur')
        
        return jsonify({
            'success': True,
            'total_records': len(raw_data),
            'columns': list(raw_data.columns),
            'unique_depots': raw_data['Name 1'].unique().tolist(),
            'sample_materials': raw_data['Material Description'].head(10).tolist()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@main.route('/search')
def search_page():
    try:
        if not session.get('file_loaded'):
            return render_template('index.html', error='Please select a depot first')
        
        return render_template('search.html')
        
    except Exception as e:
        return render_template('index.html', error=str(e))