import subprocess
# Run the script with unbuffered output so we can see it immediately
print("Running test script for 1 epoch to verify everything works...")
with open("self_pruning_network.py", "r") as f:
    code = f.read()

# temporarily change epochs to 1 for quick test
code = code.replace("range(1, 16):", "range(1, 2):")

with open("test_run.py", "w") as f:
    f.write(code)

proc = subprocess.Popen(["python", "-u", "test_run.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in iter(proc.stdout.readline, ''):
    print(line, end='')
proc.wait()
