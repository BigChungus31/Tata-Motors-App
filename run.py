import argparse
import os
import sys
from dotenv import load_dotenv
import argparse
import os
import sys
from dotenv import load_dotenv

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Load environment variables from backend/.env
env_path = os.path.join(current_dir, 'backend', '.env')
load_dotenv(env_path)

# Debug: Check if variables are loaded (remove this after testing)
print("SUPABASE_URL:", "Found" if os.getenv('NEXT_PUBLIC_SUPABASE_URL') or os.getenv('SUPABASE_URL') else "Missing")
print("SUPABASE_KEY:", "Found" if os.getenv('NEXT_PUBLIC_SUPABASE_ANON_KEY') or os.getenv('SUPABASE_ANON_KEY') else "Missing")

from server.app import create_app

# ADD THIS LINE
app = create_app()

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Load environment variables from backend/.env
env_path = os.path.join(current_dir, 'backend', '.env')
load_dotenv(env_path)

# Debug: Check if variables are loaded (remove this after testing)
print("SUPABASE_URL:", "Found" if os.getenv('NEXT_PUBLIC_SUPABASE_URL') or os.getenv('SUPABASE_URL') else "Missing")
print("SUPABASE_KEY:", "Found" if os.getenv('NEXT_PUBLIC_SUPABASE_ANON_KEY') or os.getenv('SUPABASE_ANON_KEY') else "Missing")

from server.app import create_app

def main():
    parser = argparse.ArgumentParser(description='Material Consumption Analyzer')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()
    
    try:
        print(f"Starting Material Consumption Analyzer on port {args.port}")
        print(f"Open your browser to: http://localhost:{args.port}")
        app.run(host='0.0.0.0', port=args.port, debug=args.debug)
    except Exception as e:
        print(f"Failed to start application: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()