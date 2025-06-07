import scapy.all as scapy
import argparse
import json
import subprocess
import os
import time
from datetime import datetime

# Function to log packet details
def log_packet(packet, anomalies, output_file):
    packet_info = {
        "timestamp": datetime.now().isoformat(),
        "src": packet[scapy.IP].src,
        "dst": packet[scapy.IP].dst,
        "protocol": packet[scapy.IP].proto,
        "size": len(packet)
    }
    
    # Check for anomalies
    if packet.haslayer(scapy.IP):
        if packet.haslayer(scapy.TCP):
            if packet[scapy.TCP].flags == 0x02:  # SYN without ACK
                anomalies.append({"type": "Protocol Violation", "description": "SYN without ACK", "packet": packet_info})
        if packet.haslayer(scapy.UDP):
            if packet[scapy.UDP].dport in [80, 443]:  # UDP on known TCP ports
                anomalies.append({"type": "Protocol Violation", "description": "UDP traffic on TCP port", "packet": packet_info})

    # Check for malformed packets
    if not scapy.is_valid_checksum(packet):
        anomalies.append({"type": "Malformed Packet", "description": "Invalid checksum", "packet": packet_info})

    # Log packet info
    with open(output_file, 'a') as f:
        f.write(json.dumps(packet_info) + "\n")

# Function to capture packets
def capture_packets(interface, output_file, duration, tls_decrypt):
    anomalies = []
    scapy.sniff(iface=interface, prn=lambda x: log_packet(x, anomalies, output_file), timeout=duration)
    
    # Output anomalies
    if anomalies:
        print("Anomalies detected:")
        for anomaly in anomalies:
            print(anomaly)

    # Optional TLS decryption
    if tls_decrypt:
        decrypt_tls(output_file)

# Function to decrypt TLS traffic using tshark
def decrypt_tls(output_file):
    if os.path.exists("sslkeys.log"):
        command = f"tshark -r {output_file} -Y 'ssl' -o tls.keylog_file:sslkeys.log"
        subprocess.run(command, shell=True)

# Main function to parse arguments and run the sniffer
def main():
    parser = argparse.ArgumentParser(description="Advanced Packet Sniffer")
    parser.add_argument("-i", "--interface", required=True, help="Network interface to sniff")
    parser.add_argument("-o", "--output", required=True, help="Output file for packet logs")
    parser.add_argument("-d", "--duration", type=int, default=60, help="Capture duration in seconds")
    parser.add_argument("--tls", action='store_true', help="Enable TLS decryption")
    args = parser.parse_args()

    # Start packet capture
    capture_packets(args.interface, args.output, args.duration, args.tls)

if __name__ == "__main__":
    main()
