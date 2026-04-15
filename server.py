import socket

HOST = '0.0.0.0'   # Listen on all interfaces
PORT = 5000

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(5)

print("Server started...")
print("Waiting for connection...")

conn, addr = server.accept()
print(f"Connected by {addr}")

while True:
    data = conn.recv(1024).decode()
    if not data:
        break
    print("Client:", data)

    reply = input("Server: ")
    conn.send(reply.encode())

conn.close()
server.close()
