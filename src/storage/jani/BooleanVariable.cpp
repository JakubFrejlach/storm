#include "src/storage/jani/BooleanVariable.h"

namespace storm {
    namespace jani {
        
        BooleanVariable::BooleanVariable(std::string const& name, storm::expressions::Variable const& variable, bool transient) : Variable(name, variable, transient) {
            // Intentionally left empty.
        }
        
        bool BooleanVariable::isBooleanVariable() const {
            return true;
        }
        
    }
}