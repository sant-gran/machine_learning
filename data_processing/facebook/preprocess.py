import gzip


# Read the first few lines of the compressed file to inspect the format
file_path = "/Users/santiagog/Desktop/facebook_data/facebook_combined.txt.gz"

lines =[]
with gzip.open(file_path, 'rt') as f:
    for _ in range (10):
        lines.append(f.readline().strip())

print(lines)

