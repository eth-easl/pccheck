#include "socket_work.h"

void setup_rank0_socket(const int port, int* server_fd, struct sockaddr_in* address, int N, std::vector<int> &client_sockets) {

    printf("Inside setup_rank0_socket\n");
    *server_fd = socket(AF_INET, SOCK_STREAM, 0);
    //setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt));

    address->sin_family = AF_INET;
    address->sin_addr.s_addr = INADDR_ANY;
    address->sin_port = htons(port);
    int addrlen = sizeof(address);

    // Bind the socket
    bind(*server_fd, (struct sockaddr *)address, sizeof(*address));
    listen(*server_fd, N);

    // Wait for all nodes to connect
    for (int i=0; i<N; i++) {
        printf("Get new connection for node %d\n", i);
        int new_socket = accept(*server_fd, (struct sockaddr *)address, (socklen_t *)&addrlen);
        client_sockets.push_back(new_socket);
        printf("Node %d connected!\n");
    }

}


void setup_other_socket(int* sock, struct sockaddr_in* serv_addr, const std::string &server_ip, int port) {

    // Connect to the coordinator
    *sock = socket(AF_INET, SOCK_STREAM, 0);
    serv_addr->sin_family = AF_INET;
    serv_addr->sin_port = htons(port);

    inet_pton(AF_INET, server_ip.c_str(), &(serv_addr->sin_addr));
    connect(*sock, (struct sockaddr *)serv_addr, sizeof(*serv_addr));
    printf("Connected!\n");

}

void wait_to_receive(std::vector<int>& client_sockets, int N) {

    for (auto sock: client_sockets) {
        int* iter = (int*)malloc(sizeof(int));
        read(sock, iter, 4);
    }

    for (int sock : client_sockets) {
        int val = 1;
        send(sock, &val, 4, 0);
    }

}

void send_and_wait(int* socket, int counter) {
    send(*socket, &counter, 4, 0);
    int* val = (int*)malloc(sizeof(int));
    read(*socket, val, 4);
}

void close_rank0_socket(std::vector<int>& client_sockets, int* server_fd) {

    for (int sock : client_sockets) {
        close(sock);
    }
    close(*server_fd);
}

void close_other_socket(int* sock) {
    close(*sock);

}
