#include "vec.h"
#include <iostream>

using namespace std;

int main()
{

    int n;
    cout << "Enter a number: ";
    cin >> n;

    switch (n)
    {
    case -1:
        cout << "negative one" << endl;
        break;
    case 0:
        cout << "zero" << endl;
        break;
    case 1:
        cout << "positive one" << endl;
        break;
    default:
        cout << "other value" << endl;
        break;
    }

    int max = 4000000;
    vector<int> fibonacci = gen_Fibonacci(max);
    cout << "Fibonacci sequence under 4,000,000 is" << endl;
    print_vector(fibonacci);

    test_isprime();

    test_factorize();

    test_prime_factorize();

    int layer;
    cout << "Enter layers of Pascal's triangle: ";
    cin >> layer;
    pascals_triangle(layer);

    return 0;
}