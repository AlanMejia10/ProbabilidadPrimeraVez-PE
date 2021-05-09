#include <iostream>
#include <cstdlib>
#include <armadillo>

void clear_screen();
void get_data(int& num_states, int& initial_state, int& final_state, int& num_periods, arma::fmat& mat_transition, bool first_time = true);
arma::frowvec compute_probability(const int& initial_state, const int& final_state, const int& num_periods, const arma::fmat& mat_transition);
void print_all_probabilities(const int& initial_state, const int& final_state, const arma::frowvec result);

int main(int arg, char* argv[]){
    arma::fmat mat_transition;
    int num_states, initial_state, final_state, num_periods;
    char res;
    bool first_time = true, show_results;

    do{
        clear_screen();
        std::cout << "Calculo de probabilidades de primera vez" << std::endl;
        do{
            get_data(num_states, initial_state, final_state, num_periods, mat_transition, first_time);
            clear_screen();
            arma::frowvec result = compute_probability(initial_state, final_state, num_periods, mat_transition);
            std::cout << "La probabilidad P" << initial_state << "(T" << final_state << " = " << num_periods << ") = " << result(num_periods - 1) << std::endl;
            std::cout << "Te gustaria ver las probabidades de los periodos anteriores? (s/n): ";
            std::cin >> res;
            if(res == 's') print_all_probabilities(initial_state, final_state, result);
            first_time = false;
            std::cout << "Te gustaria ingresar otro estado inicial, final y numero de periodos con la misma matriz? (s/n): ";
            std::cin >> res;
        } while (res == 's');
        std::cout << "Te gustaria introducir nuevamente la matriz de transicion? (s/n): ";
        first_time = true;
        std::cin >> res;
    }while(res == 's');
    return 0;
}

void get_data(int& num_states, int& initial_state, int& final_state, int& num_periods, arma::fmat& mat_transition, bool first_time){

    std::cout << "Ingresa el estado inicial: ";
    std::cin >> initial_state;
    std::cout << "Ingresa el estado final: ";
    std::cin >> final_state;
    std::cout << "Ingresa el numero de periodos: ";
    std::cin >> num_periods;

    if(first_time){
        std::cout << "Ingresa el numero de estados: ";
        std::cin >> num_states;
        mat_transition.zeros(num_states, num_states);
        std::cout << "Ingresa la matriz de transicion" << std::endl;

        for(int i = 0; i < static_cast<int>(mat_transition.n_rows); ++i)
            for(int j = 0; j < static_cast<int>(mat_transition.n_cols); ++j){
                std::cout << "Ingresa el elemento [" << i << ", " << j << "]: ";
                std::cin >> mat_transition(i, j);
            }
    }
}

arma::frowvec compute_probability(const int& initial_state, const int& final_state, const int& num_periods, const arma::fmat& mat_transition){

    arma::frowvec probabilities(num_periods);
    probabilities(0) = mat_transition(initial_state, final_state);
    if(num_periods == 1) return probabilities;

    for(int i = 2; i <= num_periods; ++i){
        // P^n
        arma::fmat pow_matrix = arma::powmat(mat_transition, i);

        float sum = 0;
        for(int m = 1; m <= i - 1; ++m){
            // P^(n-m)
            arma::fmat sum_mat = arma::powmat(mat_transition, i - m);
            sum += probabilities(m - 1) * sum_mat(final_state, final_state);
        }

        probabilities(i - 1) = pow_matrix(initial_state, final_state) - sum;
    }

    return probabilities;
}

void print_all_probabilities(const int& initial_state, const int& final_state, const arma::frowvec result){
    for(int i = 0; i < static_cast<int>(result.n_cols); ++i)
        std::cout << "La probabilidad P" << initial_state << "(T" << final_state << " = " << i + 1 << ") = " << result(i) << std::endl;
}

void clear_screen(){
#ifdef _WIN32
    std::system("cls");
#else
    std::system ("clear");
#endif
}
