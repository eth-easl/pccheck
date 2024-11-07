#include <iostream>
#include <thread>
#include <vector>
#include <string>
#include <cstring>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

void setup_rank0_socket(const int port, int* server_fd, struct sockaddr_in* address, int N, std::vector<int> &client_sockets);
void setup_other_socket(int* sock, struct sockaddr_in* serv_addr, const std::string &server_ip, int port);
void wait_to_receive(std::vector<int>& client_sockets, int N);
void send_and_wait(int* socket, int counter);
void close_rank0_socket(std::vector<int>& client_sockets, int* server_fd);
void close_other_socket(int* sock);