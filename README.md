# Cyber Security Projects

A comprehensive collection of cybersecurity tools, machine learning models, and educational projects designed for learning and practical application in network security, threat detection, and security analysis.

## Project Structure

### Machine Learning & Threat Detection
- **`cyber security ML/`** - Advanced machine learning models for network threat detection
  - Network intrusion detection using Random Forest classifier
  - Correlation analysis for different attack types (DDoS, Brute Force, Botnet, etc.)
  - Kaggle competition submissions for spaceship titanic dataset
  - Performance metrics and model evaluation tools

### Easy Projects
Beginner-friendly cybersecurity tools and utilities:

- **`Link checker/`** - URL Safety Analyzer
  - Heuristic analysis for malicious URL detection
  - Frontend interface for easy URL testing
  - Checks for suspicious patterns and characteristics

- **`packet sniffer/`** - Network Packet Analysis Tool
  - Real-time network packet monitoring
  - Anomaly detection for protocol violations
  - Malformed packet identification
  - JSON logging capabilities

- **`Security Tester/`** - Password Strength Evaluator
  - Flask web application for password analysis
  - Integration with Have I Been Pwned API
  - Entropy calculation and crack time estimation
  - Pattern recognition and feedback system

### Hard Projects
Advanced cybersecurity implementations:

- **`machine_learning.py`** - Advanced ML algorithms for security applications
- Custom implementations of complex security protocols

## Features

### Network Security
- **Real-time packet monitoring** with anomaly detection
- **Protocol violation detection** for TCP/UDP traffic
- **Malformed packet identification** with checksum validation

### Machine Learning Security
- **Multi-class threat classification** (Benign, Botnet, Brute Force, DDoS, DoS, Infiltration, Port Scan, Web Attack)
- **Feature correlation analysis** for different attack types
- **Model persistence** with pickle serialization
- **Performance evaluation** with detailed metrics

### Web Security
- **URL safety analysis** with heuristic scoring
- **Password strength assessment** with multiple evaluation criteria
- **Breach detection** through HIBP integration

### Data Analysis
- **Correlation matrices** for security features
- **Statistical analysis** of network traffic patterns
- **Visualization tools** for security metrics

## 🛠️ Technologies Used

- **Python** - Primary programming language
- **Scikit-learn** - Machine learning framework
- **Pandas & NumPy** - Data manipulation and analysis
- **Flask** - Web application framework
- **Scapy** - Network packet manipulation
- **BeautifulSoup** - Web scraping and HTML parsing
- **Matplotlib & Seaborn** - Data visualization
- **zxcvbn** - Password strength estimation

## Machine Learning Models

### Network Intrusion Detection
- **Algorithm**: Random Forest Classifier
- **Features**: 57 network flow characteristics
- **Classes**: 8 different threat types
- **Performance**: High accuracy with detailed classification reports

### Feature Engineering
- Flow duration and packet statistics
- Inter-arrival time analysis
- Flag count monitoring
- Packet size analysis
- Window size tracking

## Installation & Usage

### Prerequisites
```bash
pip install pandas scikit-learn flask scapy beautifulsoup4 matplotlib seaborn zxcvbn requests
```

### Running the Tools

#### Password Strength Tester
```bash
cd "Easy/Security Tester"
python Password_strenght.py
```

#### Packet Sniffer
```bash
cd "Easy/packet sniffer"
python packet_sniffer.py
```

#### URL Safety Analyzer
```bash
cd "Easy/Link checker"
python url_safety_analyzer.py
```

#### Machine Learning Model
```bash
cd "cyber security ML"
python machine_learning.py
```

## Project Highlights

- **Educational Value**: Perfect for learning cybersecurity concepts
- **Practical Applications**: Real-world security tool implementations
- **Machine Learning Integration**: Advanced AI-driven threat detection
- **Comprehensive Coverage**: From basic tools to advanced algorithms
- **Well-Documented**: Clear code structure and comments

## Learning Objectives

- Understanding network security fundamentals
- Implementing machine learning for cybersecurity
- Building web-based security tools
- Analyzing network traffic patterns
- Developing threat detection systems

## License

This project is intended for educational purposes and cybersecurity learning.

## Contributing

Feel free to contribute by:
- Adding new security tools
- Improving existing algorithms
- Enhancing documentation
- Reporting issues or suggestions

---


*Last updated: October 2025*
