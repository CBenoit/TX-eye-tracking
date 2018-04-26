#pragma once

#include "matrix.hpp"

template <typename T>
class twin_el {
public:
	twin_el(T& a, T& b) : els{a,b} {}

	twin_el& operator/=(const T& scal) {
		first() /= scal;
		second() /= scal;
		return *this;
	}

	twin_el operator/(const T& scal) const {
		twin_el cpy = *this;
		cpy /= scal;
		return cpy;
	}

	twin_el& operator*=(const T& scal) {
		first() *= scal;
		second() *= scal;
		return *this;
	}

	twin_el operator*(const T& scal) const {
		twin_el cpy = *this;
		cpy *= scal;
		return cpy;
	}

	twin_el& operator-=(const T& scal) {
		first() -= scal;
		second() -= scal;
		return *this;
	}

	twin_el operator-(const T& scal) const {
		twin_el cpy = *this;
		cpy -= scal;
		return cpy;
	}

	twin_el& operator+=(const T& scal) {
		first() += scal;
		second() += scal;
		return *this;
	}

	twin_el operator+(const T& scal) const {
		twin_el cpy = *this;
		cpy += scal;
		return cpy;
	}

	twin_el& operator=(const T& scal) {
		first() = scal;
		second() = scal;
		return *this;
	}

	twin_el& operator=(const std::pair<T, T>& scal) {
		first() = scal.first;
		second() = scal.second;
		return *this;
	}

	bool operator==(const std::pair<T, T>& p) {
		return first() == p.first && second() == p.second;
	}

	bool operator!=(const std::pair<T,T>& p) {
		return !(*this == p);
	}

	T& first() {
		return els.first;
	}

	const T& first() const {
		return els.first;
	}

	T& second() {
		return els.second;
	}

	const T& second() const {
		return els.second;
	}

private:
	std::pair<T&, T&> els;
};

template <typename T>
struct twin_matrixes {
	using el = twin_el<T>;

	twin_matrixes(const matrix<T>& m1, const matrix<T>& m2) : matrixes(m1,m2) {}

	el operator() (unsigned int x, unsigned int y) {
		return {first()(x, y), second()(x, y)};
	}

	matrix<T>& first() {
		return matrixes.first;
	}

	const matrix<T>& first() const {
		return matrixes.first;
	}

	matrix<T>& second() {
		return matrixes.second;
	}

	const matrix<T>& second() const {
		return matrixes.second;
	}

private:
	std::pair<matrix<T>, matrix<T>> matrixes;
};
