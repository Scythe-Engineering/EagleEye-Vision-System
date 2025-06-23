import socket
import pickle
import os
from time import sleep
import sys
import argparse

# Configuration
TCP_PORT = 12345       # Must match the server's TCP port
UDP_PORT = 54321       # Must match the server's UDP discovery port
DISCOVERY_MSG = "DISCOVER_SERVER"
RESPONSE_MSG = "SERVER_HERE"
BROADCAST_ADDR = '<broadcast>'  # Special address for UDP broadcast

def discover_server(timeout=3):
    """
    Send a UDP broadcast to discover the server.
    Returns the server's IP address if found, otherwise None.
    """
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    udp_sock.settimeout(timeout)

    try:
        print("[UDP] Sending discovery broadcast...")
        sys.stdout.flush()
        udp_sock.sendto(DISCOVERY_MSG.encode('utf-8'), (BROADCAST_ADDR, UDP_PORT))
        print(f"[UDP] Sent discovery message to {BROADCAST_ADDR}:{UDP_PORT}")
        sys.stdout.flush()
        data, addr = udp_sock.recvfrom(9988)
        if data.decode('utf-8') == RESPONSE_MSG:
            print(f"[UDP] Server discovered at {addr[0]}")
            sys.stdout.flush()
            return addr[0]
    except socket.timeout:
        print("[UDP] Discovery timed out. No server found.")
    except Exception as e:
        print(f"[UDP] Error during discovery: {e}")

    return None

def send_folder(folder_path, tcp_sock):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'rb') as f:
                file_data = f.read()
                file_info = pickle.dumps({
                    "file_name": os.path.relpath(file_path, folder_path).replace("\\", "/"),
                    "file_data": file_data
                })
                tcp_sock.sendall(len(file_info).to_bytes(4, 'big'))  # Send length first
                tcp_sock.sendall(file_info)  # Then send data
                print(f"[TCP] Sent file: {file_path}")
                sleep(0.05)  # Allow the server to process

    tcp_sock.sendall(b"EOF")  # End of transmission
    print("[TCP] Folder transfer complete.")

def tcp_client(server_ip, folder_path):
    """
    Connect to the server via TCP and send a folder.
    """
    tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        print(f"[TCP] Connecting to server at {server_ip}:{TCP_PORT} ...")
        tcp_sock.connect((server_ip, TCP_PORT))
        sleep(1)
        send_folder(folder_path, tcp_sock)
        print("[TCP] Folder sent successfully.")
    except Exception as e:
        print(f"[TCP] Error: {e}")
    finally:
        tcp_sock.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Client for sending a folder to a server via TCP. "
                    "Either supply a hostname/IP for a direct connection, or let the client use UDP broadcast discovery."
    )
    parser.add_argument(
        "-H", "--hostname",
        type=str,
        help="Hostname or IP address of the server for a direct connection."
    )
    parser.add_argument(
        "-f", "--folder",
        type=str,
        required=True,
        help="Path to the folder that should be sent."
    )

    args = parser.parse_args()

    folder_path = args.folder
    server_ip = args.hostname

    if server_ip:
        print(f"[INFO] Using direct connection to server at: {server_ip}")
    else:
        print("[INFO] No hostname provided. Attempting UDP discovery...")
        server_ip = discover_server()

    if server_ip:
        tcp_client(server_ip, folder_path)
    else:
        print("Server could not be discovered.")
