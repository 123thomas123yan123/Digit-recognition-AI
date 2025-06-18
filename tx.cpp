#include<iostream>
#include<cmath>
#include<random>
#include<iomanip>

constexpr const double lr = 0.001;

#define RESET   "\033[0m"
#define BLACK   "\033[30m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"
#define WHITE   "\033[37m"
#define BOLD    "\033[1m"

double relu(double x){
    return (x > 0 ? x : 0.01 * x);
}

double d_relu(double x){
    return (x > 0 ? 1 : 0.01);
}

double get_loss(double y, double y_h){
    return (0.5 * std::pow(y - y_h, 2));
}

double get_input(){
    std::random_device rd;
    std::mt19937 gen(rd()); 
    std::uniform_int_distribution<> distrib(1, 10);
    int r = distrib(gen);
    return r;
}

double get_output(double w, double b, double x){
    return (w * x + b);
}

double get_y(double x){
    return (x * 8) + 7;
}

int main(){
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);
    std::cout<<std::fixed<<std::setprecision(10);
    double b = 0;
    double w = -1.5;

    double loss;
    for(int i = 0; i <= 100000; i++){
        double x = get_input();
        double z = get_output(w, b, x);
        double a = relu(z);
        double y = get_y(x);
        loss = get_loss(y, a);
        if(i % 10000 == 0 || i < 1000){
            std::cout<<RESET<<"Run "<<RESET<<MAGENTA<<i<<":"<<(char)0x0A;
            std::cout<<RESET<<"Input: "<<x<<", output: "<<MAGENTA<<a<<RESET<<", loss: "<<(std::abs(loss) < 1 ? GREEN : (std::abs(loss) < 5 ? YELLOW : RED))<<loss<<(char)0x0A;
            std::cout<<GREEN<<"Correct output: "<<y<<(char)0x0A;
            std::cout<<RESET<<"Weight: "<<BLUE<<w<<RESET<<", bias: "<<BLUE<<b<<(char)0x0A;
        }

        double dL_da = a - y;
        double da_dz = d_relu(z);
        double dz_dw = x;
        double dz_db = 1;

        double dL_dw = dL_da * da_dz * dz_dw;
        double dL_db = dL_da * da_dz * dz_db;

        double delta_w = lr * dL_dw;
        double delta_b = lr * dL_db;

        w -= delta_w;
        b -= delta_b;
        if(i % 10000 == 0 || i < 1000){
            std::cout<<RESET<<"Change of weight: "<<CYAN<<delta_w<<RESET<<", change of bias: "<<CYAN<<delta_b<<(char)0x0A;
            std::cout<<RESET<<"Resulting weight: "<<CYAN<<w<<RESET<<", resulting bias: "<<CYAN<<b<<(char)0x0A;
        }
    }
    if(std::abs(loss) < 1){
        std::cout<<GREEN<<"Model trained succefully!"<<(char)0x0A;
        std::cout<<CYAN<<"Please input a value to test the model on the function: (x * 8 + 7)"<<RESET<<(char)0x0A;
        double input;
        while(std::cin>>input){
            float z = get_output(w, b, input);
            std::cout<<RESET<<"Result: "<<GREEN<<relu(z)<<RESET<<(char)0x0A;
        }
    } else {
        std::cout<<RED<<"Model did not train properly!"<<(char)0x0A;
        std::cout<<YELLOW<<"Try ajusting constants, increasing training, ar changing the NN!"<<(char)0x0A;
        std::cout<<MAGENTA<<"Exiting!"<<(char)0x0A;
    }
    
    return 0;
}