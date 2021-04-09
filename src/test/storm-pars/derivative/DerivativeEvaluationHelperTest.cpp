#include "test/storm_gtest.h"
#include "environment/solver/GmmxxSolverEnvironment.h"
#include "environment/solver/SolverEnvironment.h"
#include "environment/solver/TopologicalSolverEnvironment.h"
#include "solver/EliminationLinearEquationSolver.h"
#include "test/storm_gtest.h"
#include "storm-config.h"
#include "storm/api/builder.h"
#include "storm/api/storm.h"

#include "storm/storage/expressions/ExpressionManager.h"

#include "storm/adapters/RationalFunctionAdapter.h"
#include "storm/logic/Formulas.h"
#include "storm/models/sparse/StandardRewardModel.h"
#include "storm/modelchecker/prctl/SparseDtmcPrctlModelChecker.h"
#include "storm/modelchecker/results/ExplicitQuantitativeCheckResult.h"
#include "storm/storage/SparseMatrix.h"
#include "storm/storage/expressions/BinaryRelationExpression.h"

#include "carl/core/RationalFunction.h"

#include "storm-parsers/api/storm-parsers.h"
#include "storm-parsers/parser/ValueParser.h"

#include "storm-pars/api/storm-pars.h"
#include "storm-pars/transformer/SparseParametricDtmcSimplifier.h"
#include "storm-pars/analysis/OrderExtender.h"
#include "storm-pars/derivative/DerivativeEvaluationHelper.h"

template<typename ValueType>
using VariableType = typename storm::utility::parametric::VariableType<ValueType>::type;
template<typename ValueType>
using CoefficientType = typename storm::utility::parametric::CoefficientType<ValueType>::type;
template<typename ValueType>
using Instantiation = std::map<VariableType<storm::RationalFunction>, CoefficientType<storm::RationalFunction>>;
template<typename ValueType>
using ResultMap = std::map<VariableType<storm::RationalFunction>, storm::RationalNumber>;


template<typename ValueType, typename ConstantType>
std::vector<ConstantType> calculateProbability(
        std::shared_ptr<storm::models::sparse::Dtmc<storm::RationalFunction>> model,
        std::shared_ptr<const storm::logic::Formula> formulaWithoutBound,
        const std::map<VariableType<ValueType>, CoefficientType<ValueType>> &substitutions) {
    storm::modelchecker::SparseDtmcInstantiationModelChecker<storm::models::sparse::Dtmc<ValueType>, ConstantType> instantiationModelChecker(*model);
    const storm::modelchecker::CheckTask<storm::logic::Formula, ValueType> checkTask
        = storm::modelchecker::CheckTask<storm::logic::Formula, ValueType>(*formulaWithoutBound);
    instantiationModelChecker.specifyFormula(checkTask);
    storm::Environment environment;
    std::unique_ptr<storm::modelchecker::CheckResult> result = instantiationModelChecker.check(environment, substitutions);
    return result->asExplicitQuantitativeCheckResult<ConstantType>().getValueVector();
}

void testModel(std::shared_ptr<storm::models::sparse::Dtmc<storm::RationalFunction>> dtmc, std::vector<std::shared_ptr<const storm::logic::Formula>> formulas, storm::RationalFunction reachabilityFunction) {
    uint_fast64_t initialState;           
    const storm::storage::BitVector initialVector = dtmc->getStates("init");
    for (uint_fast64_t x : initialVector) {
        initialState = x;
        break;
    }

    auto formulaWithoutBound = std::make_shared<storm::logic::ProbabilityOperatorFormula>(
            formulas[0]->asProbabilityOperatorFormula().getSubformula().asSharedPointer(), storm::logic::OperatorInformation(boost::none, boost::none));

    auto parameters = storm::models::sparse::getProbabilityParameters(*dtmc);
    storm::derivative::DerivativeEvaluationHelper<storm::RationalFunction, storm::RationalNumber> helper(dtmc, parameters, formulas);

    std::map<VariableType<storm::RationalFunction>, storm::RationalFunction> derivatives;
    for (auto const& parameter : parameters) {
        derivatives[parameter] = reachabilityFunction.derivative(parameter);
    }

    // Generate test cases.
    std::vector<Instantiation<storm::RationalFunction>> testInstantiations;
    Instantiation<storm::RationalFunction> emptyInstantiation;
    testInstantiations.push_back(emptyInstantiation);
    for (auto const& param : parameters) {
        std::vector<Instantiation<storm::RationalFunction>> newInstantiations;
        for (auto point : testInstantiations) {
            for (storm::RationalNumber x = 0; x <= 1; x += .1) {
                std::map<VariableType<storm::RationalFunction>, CoefficientType<storm::RationalFunction>> newMap(point);
                newMap[param] = storm::utility::convertNumber<CoefficientType<storm::RationalFunction>>(x);
                newInstantiations.push_back(newMap);
            }
        }
        testInstantiations = newInstantiations;
    }

    // The test cases we are going to study. Left are the actual instantiations, right are the maps
    // for the results (which happen to share the same type).
    std::map<Instantiation<storm::RationalFunction>, ResultMap<storm::RationalFunction>> testCases;
    for (auto const& instantiation : testInstantiations) {
        ResultMap<storm::RationalFunction> resultMap;
        for (auto const& entry : instantiation) {
            auto parameter = entry.first;
            auto derivativeWrtParameter = derivatives[parameter];
            storm::RationalNumber evaluatedDerivative = storm::utility::convertNumber<storm::RationalNumber>(derivativeWrtParameter.evaluate(instantiation));
            resultMap[parameter] = evaluatedDerivative;
        }
        testCases[instantiation] = resultMap;
    }

    for (auto const& testCase : testCases) {
        auto instantiation = testCase.first;
        for (auto const& position : instantiation) {
            std::cout << position << std::endl;
            auto parameter = position.first;
            auto parameterValue = position.second;
            auto expectedResult = testCase.second.at(parameter);

            auto probability = calculateProbability<storm::RationalFunction, storm::RationalNumber>(dtmc, formulaWithoutBound, instantiation);
            auto derivative = helper.calculateDerivative(parameter, instantiation, probability);
            std::cout << derivative << ", " << expectedResult << std::endl;
            ASSERT_NEAR(derivative, expectedResult, 1e-6) << instantiation;
        }
    }
}

// A very simple DTMC
TEST(DerivativeEvaluationHelperTest, Simple) {
    std::string programFile = STORM_TEST_RESOURCES_DIR "/pdtmc/gradient1.pm";
    std::string formulaAsString = "Pmax=? [F s=2]";
    std::string constantsAsString = ""; //e.g. pL=0.9,TOACK=0.5

    // We have to create the dtmc and formulas here, because we need its parameters to create the polynomial
    storm::prism::Program program = storm::api::parseProgram(programFile);
    program = storm::utility::prism::preprocess(program, constantsAsString);
    std::vector<std::shared_ptr<const storm::logic::Formula>> formulas = storm::api::extractFormulasFromProperties(storm::api::parsePropertiesForPrismProgram(formulaAsString, program));
    std::shared_ptr<storm::models::sparse::Dtmc<storm::RationalFunction>> model = storm::api::buildSparseModel<storm::RationalFunction>(program, formulas)->as<storm::models::sparse::Dtmc<storm::RationalFunction>>();
    std::shared_ptr<storm::models::sparse::Dtmc<storm::RationalFunction>> dtmc = model->as<storm::models::sparse::Dtmc<storm::RationalFunction>>();
    auto simplifier = storm::transformer::SparseParametricDtmcSimplifier<storm::models::sparse::Dtmc<storm::RationalFunction>>(*dtmc);
    ASSERT_TRUE(simplifier.simplify(*(formulas[0])));
    model = simplifier.getSimplifiedModel();
    dtmc = model->as<storm::models::sparse::Dtmc<storm::RationalFunction>>();

    // The associated polynomial. In this case, it's p * (1 - p).
    carl::Variable varP; 
    for (auto parameter : storm::models::sparse::getProbabilityParameters(*dtmc)) {
        if (parameter.name() == "p") {
            varP = parameter;
        }
    }
    std::shared_ptr<storm::RawPolynomialCache> cache = std::make_shared<storm::RawPolynomialCache>();
    auto p = storm::RationalFunction(storm::Polynomial(storm::RawPolynomial(varP), cache));
    storm::RationalFunction reachabilityFunction = p * (storm::RationalFunction(1)-p);

    testModel(dtmc, formulas, reachabilityFunction);
}

// A very simple DTMC with two parameters
TEST(DerivativeEvaluationHelperTest, Simple2) {
    std::string programFile = STORM_TEST_RESOURCES_DIR "/pdtmc/gradient2.pm";
    std::string formulaAsString = "Pmax=? [F s=2]";
    std::string constantsAsString = ""; //e.g. pL=0.9,TOACK=0.5

    // We have to create the dtmc and formulas here, because we need its parameters to create the polynomial
    storm::prism::Program program = storm::api::parseProgram(programFile);
    program = storm::utility::prism::preprocess(program, constantsAsString);
    std::vector<std::shared_ptr<const storm::logic::Formula>> formulas = storm::api::extractFormulasFromProperties(storm::api::parsePropertiesForPrismProgram(formulaAsString, program));
    std::shared_ptr<storm::models::sparse::Dtmc<storm::RationalFunction>> model = storm::api::buildSparseModel<storm::RationalFunction>(program, formulas)->as<storm::models::sparse::Dtmc<storm::RationalFunction>>();
    std::shared_ptr<storm::models::sparse::Dtmc<storm::RationalFunction>> dtmc = model->as<storm::models::sparse::Dtmc<storm::RationalFunction>>();
    auto simplifier = storm::transformer::SparseParametricDtmcSimplifier<storm::models::sparse::Dtmc<storm::RationalFunction>>(*dtmc);
    ASSERT_TRUE(simplifier.simplify(*(formulas[0])));
    model = simplifier.getSimplifiedModel();
    dtmc = model->as<storm::models::sparse::Dtmc<storm::RationalFunction>>();

    // The associated polynomial. In this case, it's p * (1 - q).
    carl::Variable varP;
    carl::Variable varQ;
    for (auto parameter : storm::models::sparse::getProbabilityParameters(*dtmc)) {
        if (parameter.name() == "p") {
            varP = parameter;
        } else  if (parameter.name() == "q") {
            varQ = parameter;
        }
    }
    std::shared_ptr<storm::RawPolynomialCache> cache = std::make_shared<storm::RawPolynomialCache>();
    auto p = storm::RationalFunction(storm::Polynomial(storm::RawPolynomial(varP), cache));
    auto q = storm::RationalFunction(storm::Polynomial(storm::RawPolynomial(varQ), cache));
    storm::RationalFunction reachabilityFunction = p * (storm::RationalFunction(1) - q);

    testModel(dtmc, formulas, reachabilityFunction);
}

// The bounded retransmission protocol
TEST(DerivativeEvaluationHelperTest, Brp162) {
    std::string programFile = STORM_TEST_RESOURCES_DIR "/pdtmc/brp16_2.pm";
    std::string formulaAsString = "Pmax=? [F s=4 & i=N ]";
    std::string constantsAsString = ""; //e.g. pL=0.9,TOACK=0.5

    // We have to create the dtmc and formulas here, because we need its parameters to create the polynomial
    storm::prism::Program program = storm::api::parseProgram(programFile);
    program = storm::utility::prism::preprocess(program, constantsAsString);
    std::vector<std::shared_ptr<const storm::logic::Formula>> formulas = storm::api::extractFormulasFromProperties(storm::api::parsePropertiesForPrismProgram(formulaAsString, program));
    std::shared_ptr<storm::models::sparse::Dtmc<storm::RationalFunction>> model = storm::api::buildSparseModel<storm::RationalFunction>(program, formulas)->as<storm::models::sparse::Dtmc<storm::RationalFunction>>();
    std::shared_ptr<storm::models::sparse::Dtmc<storm::RationalFunction>> dtmc = model->as<storm::models::sparse::Dtmc<storm::RationalFunction>>();
    auto simplifier = storm::transformer::SparseParametricDtmcSimplifier<storm::models::sparse::Dtmc<storm::RationalFunction>>(*dtmc);
    ASSERT_TRUE(simplifier.simplify(*(formulas[0])));
    model = simplifier.getSimplifiedModel();
    dtmc = model->as<storm::models::sparse::Dtmc<storm::RationalFunction>>();

    carl::Variable pLVar;
    carl::Variable pKVar;
    for (auto parameter : storm::models::sparse::getProbabilityParameters(*dtmc)) {
        if (parameter.name() == "pL") {
            pLVar = parameter;
        } else  if (parameter.name() == "pK") {
            pKVar = parameter;
        }
    }
    std::shared_ptr<storm::RawPolynomialCache> cache = std::make_shared<storm::RawPolynomialCache>();
    auto pL = storm::RationalFunction(storm::Polynomial(storm::RawPolynomial(pLVar), cache));
    auto pK = storm::RationalFunction(storm::Polynomial(storm::RawPolynomial(pKVar), cache));

    // The term is ((pK)^16 * (pL)^16 * (pK^2*pL^2+(-3)*pK*pL+3)^16)/(1), so we're just going to create this here.
    // I'm sorry. There is no ^ operator.
    auto firstTerm = pK * pK * pK * pK * pK * pK * pK * pK * pK * pK * pK * pK * pK * pK * pK * pK; 
    auto secondTerm = pL * pL * pL * pL * pL * pL * pL * pL * pL * pL * pL * pL * pL * pL * pL * pL; 
    auto thirdTermUnpowed = pK*pK*pL*pL+(-3)*pK*pL+3;
    auto thirdTerm = thirdTermUnpowed * thirdTermUnpowed * thirdTermUnpowed * thirdTermUnpowed * thirdTermUnpowed * thirdTermUnpowed * thirdTermUnpowed * thirdTermUnpowed * thirdTermUnpowed * thirdTermUnpowed * thirdTermUnpowed * thirdTermUnpowed * thirdTermUnpowed * thirdTermUnpowed * thirdTermUnpowed * thirdTermUnpowed;
    storm::RationalFunction reachabilityFunction = firstTerm * secondTerm * thirdTerm;

    testModel(dtmc, formulas, reachabilityFunction);
}
