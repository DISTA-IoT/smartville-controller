from flask import Flask, jsonify
import subprocess
import socket
import threading
import os
import time

def setup_port_forwarding(external_port, internal_port):
    """
    Sets up port forwarding from container's external IP:port to localhost:internal_port
    Works with both socat (if available) or falls back to Python socket implementation
    """
    print(f"Setting up port forwarding: external port {external_port} → internal port {internal_port}")
    
    # First check if socat is installed
    try:
        subprocess.run(["which", "socat"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Socat is installed, using it for port forwarding...")
        
        # Use socat for forwarding (most efficient option)
        cmd = f"socat TCP-LISTEN:{external_port},fork,reuseaddr TCP:127.0.0.1:{internal_port} &"
        subprocess.Popen(cmd, shell=True)
        print(f"Port forwarding set up with socat: {external_port} → {internal_port}")
        return True
        
    except subprocess.CalledProcessError:
        print("Socat not installed, attempting to install it...")
        try:
            # Try to install socat
            subprocess.run("apt-get update -qq && apt-get install -y socat", 
                          shell=True, check=True, stdout=subprocess.PIPE)
            
            # Now use socat
            cmd = f"socat TCP-LISTEN:{external_port},fork,reuseaddr TCP:127.0.0.1:{internal_port} &"
            subprocess.Popen(cmd, shell=True)
            print(f"Port forwarding set up with socat: {external_port} → {internal_port}")
            return True
            
        except subprocess.CalledProcessError:
            print("Couldn't install socat, using Python socket forwarding...")
            
            # Fall back to Python socket implementation
            def forward_server():
                server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind(('0.0.0.0', external_port))
                server.listen(5)
                
                print(f"Python forwarding set up: {external_port} → {internal_port}")
                
                while True:
                    try:
                        client_sock, client_addr = server.accept()
                        print(f"New connection from {client_addr}")
                        
                        target = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        target.connect(('127.0.0.1', internal_port))
                        
                        # Start bidirectional forwarding
                        threading.Thread(target=forward, args=(client_sock, target), daemon=True).start()
                        threading.Thread(target=forward, args=(target, client_sock), daemon=True).start()
                        
                    except Exception as e:
                        print(f"Error in forwarding: {e}")
            
            # Helper function to forward data between sockets
            def forward(source, destination):
                try:
                    while True:
                        data = source.recv(4096)
                        if not data:
                            break
                        destination.send(data)
                except:
                    pass
                finally:
                    try:
                        source.close()
                    except:
                        pass
                    try:
                        destination.close()
                    except:
                        pass
            
            # Start forwarding in a background thread
            forwarding_thread = threading.Thread(target=forward_server, daemon=True)
            forwarding_thread.start()
            return True

# Create a simple Flask application
app = Flask(__name__)

@app.route('/')
def index():
    container_ip = get_container_ip()
    return f"""
    <h1>Hello from Flask in Container!</h1>
    <p>This Flask server is running inside a Docker container in GNS3.</p>
    <p>Container IP: {container_ip}</p>
    <p>You're accessing this through port forwarding!</p>
    """

@app.route('/api/status')
def status():
    return jsonify({
        "status": "running",
        "container_ip": get_container_ip(),
        "timestamp": time.time()
    })

def get_container_ip():
    """Get the container's IP address"""
    try:
        # This works for most Docker containers
        result = subprocess.run("hostname -I | awk '{print $1}'", 
                               shell=True, capture_output=True, text=True)
        return result.stdout.strip()
    except:
        return "Unknown"

if __name__ == "__main__":
    # The external port that will be accessible from outside the container
    EXTERNAL_PORT = 8080
    
    # The internal port where Flask will actually run
    INTERNAL_PORT = 5000
    
    # Setup port forwarding before starting Flask
    setup_port_forwarding(EXTERNAL_PORT, INTERNAL_PORT)
    
    print(f"Container IP: {get_container_ip()}")
    print(f"Access the server at http://{get_container_ip()}:{EXTERNAL_PORT}")
    
    # Start Flask on localhost only (for security)
    # The port forwarding will make it accessible externally
    app.run(host='127.0.0.1', port=INTERNAL_PORT)