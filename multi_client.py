from fl_client import FederatedClient
import datasource
import multiprocessing
import threading

server_host = "127.0.0.1"
server_port = 5000

def start_client():
    print("start client")
    c = FederatedClient(server_host, server_port, datasource.Mnist)


if __name__ == '__main__':
    jobs = []
    for i in range(20):
        # threading.Thread(target=start_client).start()

        p = multiprocessing.Process(target=start_client)
        jobs.append(p)
        p.start()
    # TODO: randomly kill