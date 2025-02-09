#include "vec.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>

using namespace std;

// Definition of functions

void print_vector(vector<int> v){
    for (int comp : v){
        cout << comp << " ";
    }
    cout << endl;
}

vector<int> gen_Fibonacci(int max){

    vector<int> F_sequence;
    int a1 = 1, a2 = 2;
    F_sequence.push_back(a1);

    int a3 = 0;
    while (a2 <= max){
        F_sequence.push_back(a2);
        a3 = a1 + a2;
        a1 = a2;
        a2 = a3;
    }
    return F_sequence;
}

bool is_prime(int n){
    // 1 is not prime number
    if (n == 1){
        return false;
    }

    int factor = 2;
    // Only needs to find below sqrt(n)
    while (factor <= pow(n, 0.5)){
        if (n % factor == 0){
            return false;
        }
        else{
            factor += 1;
        }
        
    }
    return true;
}

void test_isprime() {
    cout << "Isprime test" << endl;
    cout << "isprime(2) = " << is_prime(2) << '\n';
    cout << "isprime(10) = " << is_prime(10) << '\n';
    cout << "isprime(17) = " << is_prime(17) << '\n';
}

vector<int> factorize(int n){
    vector<int> factor_vec;
    // Find factor couple only needs below sqrt(n), save time
    for (int factor = 1; factor < pow(n, 0.5); ++factor){
        if (n % factor == 0){
            factor_vec.push_back(factor);
            factor_vec.push_back(n / factor);
        }
        
    }
    // Sort the factor vector
    sort(factor_vec.begin(), factor_vec.end());

    return factor_vec;
}

void test_factorize() {
    cout << "Factorization test" << endl;
    print_vector(factorize(2));
    print_vector(factorize(72));
    print_vector(factorize(196));
}

vector<int> prime_factorize(int n){
    vector<int> factor_vec = factorize(n);

    for (auto it = factor_vec.begin(); it != factor_vec.end(); ){
        // erase method will return next vaild iterator.
        if (is_prime(*it) == false){
            it = factor_vec.erase(it);
        }
        // If put this in the for loop statement, then after the final iterator is erased, ++it will return a wild pointer, causing Segmentation fault
        else{
            ++it;
        }
    }
    
    return factor_vec;
}

void test_prime_factorize() {
    cout << "Prime factorization test" << endl;
    print_vector(prime_factorize(2));
    print_vector(prime_factorize(72));
    print_vector(prime_factorize(196));
}

void pascals_triangle(int n){
    vector<int> row;
    vector<int> last_row;

    for (int i = 0; i < n; ++i) {
        // Resize next row
        row.resize(i + 1);

        row[0] = 1;
        row[i] = 1;

        // Calculate current row from last row
        for (int j = 1; j < i; ++j) {
            row[j] = last_row[j - 1] + last_row[j];
        }
        last_row = row;
        for (int num : row) {
            cout << num << " ";
        }
        cout << endl;
    }
}