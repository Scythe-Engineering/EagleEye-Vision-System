import socket
import sys
import threading
import pickle
import os
import shutil
import subprocess

# set working directory to current directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configuration
TCP_PORT = 12345       # Port for TCP connections (data exchange)
UDP_PORT = 54321       # Port for UDP discovery messages
DISCOVERY_MSG = "DISCOVER_SERVER"
RESPONSE_MSG = "SERVER_HERE"
RECEIVE_DIR = os.path.expanduser("../src")  # Directory to save received files

def sys_print(msg):
    print(msg)
    sys.stdout.flush()

def restart_vision():
    try:
        subprocess.run(["systemctl", "restart", "EagleEye"], check=True)
        sys_print("EagleEye restarted successfully.")
    except subprocess.CalledProcessError as e:
        sys_print(f"Failed to restart EagleEye: {e}")


def udp_discovery_listener():
    """
    Listen for UDP discovery messages and reply with a response.
    """
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Allow reuse of addresses
    udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    udp_sock.bind(('', UDP_PORT))
    sys_print(f"[UDP] Discovery listener started on port {UDP_PORT}")

    while True:
        try:
            data, addr = udp_sock.recvfrom(9988)
            message = data.decode('utf-8')
            if message == DISCOVERY_MSG:
                sys_print(f"[UDP] Discovery request received from {addr}")
                udp_sock.sendto(RESPONSE_MSG.encode('utf-8'), addr)
        except Exception as e:
            sys_print(f"[UDP] Error: {e}")

def remove_and_create_folder(target_dir):
    """
    Remove the existing folder (if any) and create a fresh folder.
    """
    if os.path.exists(target_dir):
        sys_print(f"[TCP] Removing existing folder: {target_dir}")
        shutil.rmtree(target_dir)  # Remove the folder and its contents
    os.makedirs(target_dir)  # Create the folder again
    sys_print(f"[TCP] Created folder: {target_dir}")

def save_file(file_info):
    """
    Save received file data to the server's filesystem.
    """
    file_name = file_info["file_name"]
    file_data = file_info["file_data"]
    file_path = os.path.join(RECEIVE_DIR, file_name)

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        f.write(file_data)
    sys_print(f"[TCP] Received and saved file: {file_path}")

def handle_tcp_connection(conn, addr):
    try:
        remove_and_create_folder(RECEIVE_DIR)
        while True:
            length_data = conn.recv(4)
            if not length_data:
                break
            if length_data == b"EOF":
                sys_print("[TCP] Folder transfer complete.")
                restart_vision()
                break
            length = int.from_bytes(length_data, 'big')
            data = b""
            while len(data) < length:
                packet = conn.recv(length - len(data))
                if not packet:
                    raise ValueError("Incomplete data received")
                data += packet
            file_info = pickle.loads(data)
            save_file(file_info)
    except Exception as e:
        sys_print(f"[TCP] Error processing data: {e}")
    finally:
        conn.close()
        sys_print(f"[TCP] Connection closed with {addr}")

def tcp_server():
    tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    tcp_sock.bind(('', TCP_PORT))
    tcp_sock.listen(1)
    sys_print(f"[TCP] Server listening on port {TCP_PORT}")

    while True:
        conn, addr = tcp_sock.accept()
        sys_print(f"[TCP] Connection accepted from {addr}")
        handle_tcp_connection(conn, addr)

if __name__ == '__main__':
    # Start UDP discovery listener in a background thread
    udp_thread = threading.Thread(target=udp_discovery_listener, daemon=True)
    udp_thread.start()

    # Start TCP server (runs in the main thread)
    tcp_server()
