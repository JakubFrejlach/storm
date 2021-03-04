#ifndef STORM_DERIVATIVEGRADIENTDESCENTMETHOD_H
#define STORM_DERIVATIVEGRADIENTDESCENTMETHOD_H
#include <boost/optional.hpp>
#include <string>
namespace storm {
	namespace derivative {
		/**
		 * GradientDescentMethod is the method of Gradient Descent the GradientDescentInstantiationSearcher
		 * shall use. 
		 */
		enum class GradientDescentMethod {
			ADAM, ///< The default.
			RMSPROP,
			PLAIN,
			PLAIN_SIGN,
			MOMENTUM,
			MOMENTUM_SIGN,
			NESTEROV,
			NESTEROV_SIGN
		};

	}
}
#endif
