#include <iomanip>
#include <iostream>
#include <vector>

int main() {
	std::ios_base::sync_with_stdio(false);
	std::cin.tie(nullptr);

	int n;
	std::cin >> n;

	std::cout << std::scientific << std::setprecision(10);

	std::vector<double> arr(n);

	for (int i = 0; i < n; ++i) {
		std::cin >> arr[i];
	}

	for (int i = 0; i < n; ++i) {
		double val;
		std::cin >> val;
		arr[i] += val;
	}

	for (int i = 0; i < n; ++i) {
		std::cout << arr[i] << (i == n - 1 ? "" : " ");
	}
	std::cout << std::endl;

	return 0;

}
