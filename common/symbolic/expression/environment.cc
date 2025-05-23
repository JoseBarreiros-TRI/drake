/* clang-format off to disable clang-format-includes */
// NOLINTNEXTLINE(build/include): Our header file is included by all.h.
#include "drake/common/symbolic/expression/all.h"
/* clang-format on */

#include <cmath>
#include <initializer_list>
#include <ostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <fmt/format.h>

namespace drake {
namespace symbolic {

using std::endl;
using std::initializer_list;
using std::ostream;
using std::ostringstream;
using std::runtime_error;
using std::string;

namespace {
void throw_if_nan(const double v) {
  if (std::isnan(v)) {
    ostringstream oss;
    oss << "NaN is detected in the initialization of an environment.";
    throw runtime_error(oss.str());
  }
}

// Given a list of variables, @p vars, builds an Environment::map which maps a
// Variable to its double value. All values are set to 0.0.
Environment::map BuildMap(const initializer_list<Environment::key_type> vars) {
  Environment::map m;
  for (const Environment::key_type& var : vars) {
    m.emplace(var, 0.0);
  }
  return m;
}

}  // namespace

Environment::Environment(const std::initializer_list<value_type> init)
    : Environment{map(init)} {}

Environment::Environment(const std::initializer_list<key_type> vars)
    : Environment{BuildMap(vars)} {}

Environment::Environment(map m) : map_{std::move(m)} {
  for (const auto& p : map_) {
    throw_if_nan(p.second);
  }
}

Environment::~Environment() = default;

void Environment::insert(const key_type& key, const mapped_type& elem) {
  throw_if_nan(elem);
  map_.emplace(key, elem);
}

void Environment::insert(
    const Eigen::Ref<const MatrixX<key_type>>& keys,
    const Eigen::Ref<const MatrixX<mapped_type>>& elements) {
  if (keys.rows() != elements.rows() || keys.cols() != elements.cols()) {
    throw runtime_error(fmt::format(
        "symbolic::Environment::insert: The size of keys ({} x {}) "
        "does not match the size of elements ({} x {}).",
        keys.rows(), keys.cols(), elements.rows(), elements.cols()));
  }

  for (Eigen::Index i = 0; i < keys.cols(); ++i) {
    for (Eigen::Index j = 0; j < keys.rows(); ++j) {
      insert(keys(j, i), elements(j, i));
    }
  }
}

Variables Environment::domain() const {
  Variables dom;
  for (const auto& p : map_) {
    dom += p.first;
  }
  return dom;
}

string Environment::to_string() const {
  ostringstream oss;
  oss << *this;
  return oss.str();
}

Environment::mapped_type& Environment::operator[](const key_type& key) {
  return map_[key];
}

const Environment::mapped_type& Environment::operator[](
    const key_type& key) const {
  if (!map_.contains(key)) {
    ostringstream oss;
    oss << "Environment::operator[] was called on a const Environment "
        << "with a missing key \"" << key << "\".";
    throw runtime_error(oss.str());
  }
  return map_.at(key);
}

ostream& operator<<(ostream& os, const Environment& env) {
  for (const auto& p : env) {
    os << p.first << " -> " << p.second << endl;
  }
  return os;
}

Environment PopulateRandomVariables(Environment env, const Variables& variables,
                                    RandomGenerator* const random_generator) {
  DRAKE_DEMAND(random_generator != nullptr);
  for (const Variable& var : variables) {
    const auto it = env.find(var);
    if (it != env.end()) {
      // The variable is already assigned by env, no need to sample.
      continue;
    }
    switch (var.get_type()) {
      case Variable::Type::CONTINUOUS:
      case Variable::Type::BINARY:
      case Variable::Type::BOOLEAN:
      case Variable::Type::INTEGER:
        // Do nothing for non-random variables.
        break;
      case Variable::Type::RANDOM_UNIFORM:
        env.insert(var, std::uniform_real_distribution<double>{
                            0.0, 1.0}(*random_generator));
        break;
      case Variable::Type::RANDOM_GAUSSIAN:
        env.insert(
            var, std::normal_distribution<double>{0.0, 1.0}(*random_generator));
        break;
      case Variable::Type::RANDOM_EXPONENTIAL:
        env.insert(
            var, std::exponential_distribution<double>{1.0}(*random_generator));
        break;
    }
  }
  return env;
}

}  // namespace symbolic
}  // namespace drake
