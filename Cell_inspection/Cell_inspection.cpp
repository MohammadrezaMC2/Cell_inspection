#include <cstdio>
#include <vector>
#include "spline.h"

int main(int, char**) {
	std::vector<double> X = { 0.1, 0.4, 1.2, 1.8, 2.0 }; // must be increasing
	std::vector<double> Y = { 0.1, 0.7, 0.6, 1.1, 0.9 };

	tk::spline s(X, Y);
	double x = 1.5, y = s(x), deriv = s.deriv(1, x);

	printf("spline at %f is %f with derivative %f\n", x, y, deriv);
}