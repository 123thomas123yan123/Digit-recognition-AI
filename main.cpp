#include<iostream>
#include<vector>
#include<random>
#include<algorithm>
#include<fstream>
#include<cmath>
#include<iomanip>

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

#define IMG_PATH "archive/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte"
#define LABLE_PATH "archive/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte"

constexpr const float e = 2.71828182;

namespace util{

    float lrelu(const float& z){
        return (z <= 0 ? 0.01f * z : z);
    }

    float lrelu_d(const float& z){
        return (z <= 0 ? 0.01f : 1);
    }

    float loss(const float& y, const float& yh){
        return 0.5 * (y - yh) * (y - yh);
    }

    float cross_entropy_loss(const std::vector<float>& tl, const std::vector<float>& ap){
        int C = tl.size();
        float sum = 0;
        for(int i = 0; i < C; i++){
            sum += tl[i] * std::log2(ap[i]);
        }
        return sum;
    }

    std::vector<float> softmax(std::vector<float> input){
        int n = input.size();
        float sum = 0;
        for(int i = 0; i < n; i++){
            input[i] = std::exp(input[i]);
            sum += input[i];
        }
        for(int i = 0; i < n; i++){
            input[i] /= sum;
        }
        return input;
    }

    std::pair<std::mt19937, std::normal_distribution<float> > rnd(){
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0, 1);
        return {gen, dist};
    }

    uint32_t read_big_endian_uint32(std::ifstream& stream){
        uint8_t bytes[4];
        stream.read(reinterpret_cast<char*>(bytes), sizeof(bytes));
        return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
    }

    std::vector<std::vector<float> > load_mnist_images(const std::string& filename){
        std::ifstream file(filename, std::ios::binary);
        if(!file) throw std::runtime_error("Failed to open MNIST images: " + filename);

        uint32_t magic = read_big_endian_uint32(file);

        if(magic == 0xCAFEBABE) throw std::runtime_error("Dont try to give me executable, i like coffee --Tommy");
        if(magic != 0x803) throw std::runtime_error("Invalid MINIST file: " + filename);
        
        uint32_t ic = read_big_endian_uint32(file);
        uint32_t rc = read_big_endian_uint32(file);
        uint32_t cn = read_big_endian_uint32(file);

        std::vector<std::vector<float> > images(ic, std::vector<float>(rc * cn));

        for(uint32_t i = 0; i < ic; i++){
            for(int32_t j = 0; j < rc * cn; j++){
                uint8_t p = 0;
                file.read(reinterpret_cast<char* >(&p), sizeof(p));
                images[i][j] = p / 255.f;
            }
        }
        return images;
    }

    std::vector<int> load_image_labels(const std::string& filename){
        std::ifstream file(filename, std::ios::binary);
        if(!file) throw std::runtime_error("Failed to open MNIST images: " + filename);
        uint32_t magic = read_big_endian_uint32(file);
        if(magic == 0x7F454C46) throw std::runtime_error("Dont try to give me executable, i also dont take santa's helpers --Tommy");
        if(magic != 0x801) throw std::runtime_error("Invalid MINIST file: " + filename);

        uint32_t lc = read_big_endian_uint32(file);
        std::vector<int> labels(lc);
        for(uint32_t i = 0; i < lc; i++){
            uint8_t p = 0;
            file.read(reinterpret_cast<char* >(&p), sizeof(p));
            labels[i] = p;
        }
        return labels;
    }

}

struct ilayer{
    int input_size;
    int output_size;
    std::vector<float> outputs;
    std::vector<float> inputs;

    ilayer(int i_s, int o_s){
        input_size = i_s;
        output_size = o_s;
        inputs.resize(i_s);
        outputs.resize(o_s);
    }
    void forward(){
        this->outputs = this->inputs;
    }
};

struct olayer{
    int input_size;
    int output_size;
    std::vector<float> inputs;
    std::vector<float> outputs;
    std::vector<std::vector<float> > weights;
    std::vector<float> biases;
    std::vector<float> z;

    olayer(int i_s, int o_s){
        input_size = i_s;
        output_size = o_s;
        inputs.resize(i_s);
        outputs.resize(o_s);
        weights.resize(o_s);
        biases.resize(o_s, 0);
        z.resize(o_s);

        auto[gen, dist] = util::rnd();

        for(int i = 0; i < o_s; i++){
            weights[i].resize(i_s);
            for(int j = 0; j < i_s; j++){
                weights[i][j] = dist(gen) * 0.01f;
            }
        }
    }

    void forward(){
        for(int i = 0; i < output_size; i++){
            z[i] = 0;
            for(int j = 0; j < input_size; j++){
                z[i] += weights[i][j] * inputs[j];
            }
            z[i] += biases[i];
        }
        outputs = util::softmax(z);
    }

    std::pair<float, std::vector<float> > backward(float lr, const std::vector<float>& dL_dZ, const std::vector<float>& tl){
        for(int i = 0; i < output_size; i++){
            for(int j = 0; j < input_size; j++){
                weights[i][j] -= lr * dL_dZ[i] * inputs[j];
            }
            biases[i] -= lr * dL_dZ[i];
        }
        float loss = util::cross_entropy_loss(tl, outputs);
        return {loss, outputs};
    }
};

struct hlayer{
    int input_size;
    int output_size;
    std::vector<std::vector<float> > weights;
    std::vector<float> biases;
    std::vector<float> outputs;
    std::vector<float> inputs;
    std::vector<float> z;

    hlayer(int i_s, int o_s){
        input_size = i_s;
        output_size = o_s;
        inputs.resize(i_s);
        outputs.resize(o_s, 0);
        biases.resize(o_s);
        weights.resize(o_s);

        auto[gen, dist] = util::rnd();

        for(int i = 0; i < o_s; i++){
            weights[i].resize(i_s);
            for(int j = 0; j < i_s; j++){
                weights[i][j] = dist(gen) * 0.01f;
            }
        }
        z.resize(o_s);
    }

    void forward(){
        for(int i = 0; i < output_size; i++){
            z[i] = 0;
            for(int j = 0; j < input_size; j++){
                z[i] += weights[i][j] * inputs[j];
            }
            z[i] += biases[i];
            outputs[i] = util::lrelu(z[i]);
        }
    }

    void backward(float lr, std::vector<float>& dL_dz){
        for(int i = 0; i < output_size; i++){
            for(int j = 0; j < input_size; j++){
                weights[i][j] -= lr * dL_dz[i] * inputs[j];
            }
            biases[i] -= lr * dL_dz[i];
        }
    }
};

class NeuralNetwork{
    private:
        float learning_rate;
        int epochs;
        int output_interval;
        bool ios;
        ilayer input = ilayer(784, 128);
        hlayer hidden = hlayer(128, 128);
        olayer output = olayer(128, 10);
    public:
        NeuralNetwork(float lr, float ep, int oi, bool s){
            std::cout<<std::fixed<<std::setprecision(5);
            learning_rate = lr;
            epochs = ep;
            output_interval = oi;
            ios = s;
            if(ios){
                std::ios_base::sync_with_stdio(false);
                std::cin.tie(NULL);
            }
        }

        float train(){
            float loss;
            std::cout<<CYAN<<"Loading images..."<<(char)0x0A;
            std::vector<std::vector<float> > images = util::load_mnist_images(IMG_PATH);
            std::cout<<CYAN<<"Loading labels..."<<(char)0x0A;
            std::vector<int> labels = util::load_image_labels(LABLE_PATH);
            int image_count = images.size();
            std::cout<<CYAN<<"Initalizing dataset..."<<(char)0x0A;
            std::vector<std::pair<std::vector<float>, int> > dataset(image_count);
            for(int i = 0; i < image_count; i++){
                dataset[i] = {images[i], labels[i]};
            }
            std::cout<<GREEN<<"Training started!"<<(char)0x0A;
            for(int i = 0; i < epochs; i++){
                std::cout<<RESET<<"Epoch: "<<MAGENTA<<i<<(char)0x0A;
                auto [gen, dist] = util::rnd();
                std::shuffle(dataset.begin(), dataset.end(), gen);
                float total_loss = 0.0f;
                for(int j = 0; j < image_count; j++){
                    input.inputs = dataset[j].first;
                    input.outputs = input.inputs;
                    hidden.inputs = input.outputs;
                    hidden.forward();
                    output.inputs = hidden.outputs;
                    output.forward();
                    std::vector<float> tl(10, 0.0f);
                    tl[dataset[j].second] = 1.0f;
                    std::vector<float> dL_dZ_o(output.output_size);
                    for(int i = 0; i < output.output_size; i++){
                        dL_dZ_o[i] = output.outputs[i] - tl[i];
                    }
                    std::pair<float, std::vector<float> > out = output.backward(learning_rate, dL_dZ_o, tl);
                    total_loss += out.first;
                    std::vector<float> dL_dZ_h(128, 0.0f);
                    for(int i = 0; i < 10; i++){
                        for(int j = 0; j < 128; j++){
                            dL_dZ_h[j] += dL_dZ_o[i] * output.weights[i][j];
                        }
                    }
                    for(int j = 0; j < 128; j++){
                        dL_dZ_h[j] *= util::lrelu_d(hidden.z[j]);
                    }
                    hidden.backward(learning_rate, dL_dZ_h);

                    if(!(j % output_interval)){
                        std::cout<<RESET<<GREEN<<j<<"th image, loss: "<<(out.first > -0.51 ? GREEN : (out.first > 0.63 ? YELLOW : RED))<<out.first<<(char)0x0A;
                        std::cout<<RESET<<"Probablities: ";
                        for(int i = 0; i <= 9; i++){
                            std::cout<<RESET<<i<<": "<<GREEN<<out.second[i]<<"%"<<(char)0x0A;
                        }
                    }

                }
            }
            return loss;
        }
        void test(){
            std::vector<float> image(784);
            std::cout<<GREEN<<"Model trained successfully!"<<(char)0x0A;
            std::cout<<BLUE<<"Input 784 floats representing a grayscale image to test the model!"<<RESET<<(char)0x0A;
            for(int i = 0; i < 784; i++){
                std::cin>>image[i];
            }
            input.inputs = image;
            input.outputs = input.inputs;
            hidden.inputs = input.outputs;
            hidden.forward();
            output.inputs = hidden.outputs;
            output.forward();
            std::vector<float> p(10);
            p = util::softmax(output.outputs);
            std::cout<<RESET<<"Probablities: "<<(char)0x0A;
            int mm;
            float max = -1.f;
            for(int i = 0; i <= 9; i++){
                if(p[i] > max){
                    max = p[i];
                    mm = i;
                }
                std::cout<<RESET<<i<<": "<<GREEN<<p[i]<<(char)0x0A;
            }
            std::cout<<RESET<<"Predicted number: "<<GREEN<<mm<<(char)0x0A;
        }
};


int main(){
    NeuralNetwork nn(0.001, 15, 5000, true);
    nn.train();
    nn.test();
    return 0;
}
