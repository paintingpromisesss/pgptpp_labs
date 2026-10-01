#include <iomanip>
#include <iostream>
#include <cmath>

int main() {
	float a, b, c;
	std::cin >> a >> b >> c;

	std::cout << std::fixed << std::setprecision(6);

	if (a == 0.0f) {
		if (b == 0.0f) {
			if (c == 0.0f) {
				std::cout << "any" << std::endl;
			} else {
				std::cout << "incorrect" << std::endl;
			}
		} else {
			float x = -c / b;
			if (x == -0.0f) {
				x = 0.0f;
			}
			std::cout << x << std::endl;
		}
	} else {
		float D = b * b - 4.0f * a * c;

		if (D > 0.0f) {
			float sqrtD = std::sqrt(D);
			float x1 = (-b + sqrtD) / (2.0f * a);
			float x2 = (-b - sqrtD) / (2.0f * a);

			if (x1 == -0.0f) {
				x1 = 0.0f;
			}

			if (x2 == -0.0f) {
				x2 = 0.0f;
			}

			std::cout << x1 << " " << x2 << std::endl;
		} else if (D == 0.0f) {
			float x = -b / (2.0f * a);
			if (x == -0.0f) {
				x = 0.0f;
			}

			std::cout << x << std::endl;
		} else {
			std::cout << "imaginary" << std::endl;
		}
	}

	return 0;
}
