# Analyze a cryptographic challenge
python ctf_solver.py --mode crypto --text "Uryyb, jbeyq!"

# Analyze an image for steganography
python ctf_solver.py --mode stego --file suspicious.png

# Analyze a binary file for vulnerabilities
python ctf_solver.py --mode binary --file challenge.bin

# Analyze a web request for vulnerabilities
python ctf_solver.py --mode web --text "GET /search.php?q=test';DROP%20TABLE%20users;--"
