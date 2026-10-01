#include <ios>
#include <iostream>
#include <cstdlib>
#include <iomanip>

int main() {
	int n;
	std::cin >> n;

	float *arr = (float*)std::malloc(n * sizeof(float));
	for (int i = 0; i < n; ++i) {
		std::cin >> arr[i];
	}

	for (int i = 0; i < n - 1; ++i) {
		for (int j = 0; j < n - 1 - i; ++j) {
			if (arr[j] > arr[j+1]) {
				float temp = arr[j];
				arr[j] = arr[j+1];
				arr[j+1] = temp;
			}
		}
	}

	std::cout << std::scientific << std::setprecision(6);
	for (int i = 0; i < n; ++i) {
		if (arr[i] == -0.0f) {
			arr[i] = 0.0f;
		}
		std::cout << arr[i] << (i == n - 1 ? "" : " ");
	}
	std::cout << std::endl;
	std::free(arr);

	return 0;
}
