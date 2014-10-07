#include "src/settings/modules/GeneralSettings.h"

#include "src/settings/SettingsManager.h"
#include "src/settings/SettingMemento.h"
#include "src/exceptions/InvalidSettingsException.h"

namespace storm {
    namespace settings {
        namespace modules {
            
            const std::string GeneralSettings::moduleName = "general";
            const std::string GeneralSettings::helpOptionName = "help";
            const std::string GeneralSettings::helpOptionShortName = "h";
            const std::string GeneralSettings::versionOptionName = "version";
            const std::string GeneralSettings::verboseOptionName = "verbose";
            const std::string GeneralSettings::verboseOptionShortName = "v";
            const std::string GeneralSettings::precisionOptionName = "precision";
            const std::string GeneralSettings::precisionOptionShortName = "p";
            const std::string GeneralSettings::exportDotOptionName = "exportdot";
            const std::string GeneralSettings::configOptionName = "config";
            const std::string GeneralSettings::configOptionShortName = "c";
            const std::string GeneralSettings::explicitOptionName = "explicit";
            const std::string GeneralSettings::explicitOptionShortName = "e";
            const std::string GeneralSettings::symbolicOptionName = "symbolic";
            const std::string GeneralSettings::symbolicOptionShortName = "s";
            const std::string GeneralSettings::pctlOptionName = "pctl";
            const std::string GeneralSettings::pctlFileOptionName = "pctlfile";
            const std::string GeneralSettings::cslOptionName = "csl";
            const std::string GeneralSettings::cslFileOptionName = "cslfile";
            const std::string GeneralSettings::ltlOptionName = "ltl";
            const std::string GeneralSettings::ltlFileOptionName = "ltlfile";
            const std::string GeneralSettings::transitionRewardsOptionName = "transrew";
            const std::string GeneralSettings::stateRewardsOptionName = "staterew";
            const std::string GeneralSettings::counterexampleOptionName = "counterexample";
            const std::string GeneralSettings::counterexampleOptionShortName = "cex";
            const std::string GeneralSettings::dontFixDeadlockOptionName = "nofixdl";
            const std::string GeneralSettings::dontFixDeadlockOptionShortName = "ndl";
            const std::string GeneralSettings::timeoutOptionName = "timeout";
            const std::string GeneralSettings::timeoutOptionShortName = "t";
            const std::string GeneralSettings::eqSolverOptionName = "eqsolver";
            const std::string GeneralSettings::lpSolverOptionName = "lpsolver";
            const std::string GeneralSettings::constantsOptionName = "constants";
            const std::string GeneralSettings::constantsOptionShortName = "const";
            const std::string GeneralSettings::statisticsOptionName = "statistics";
            const std::string GeneralSettings::statisticsOptionShortName = "stats";

            GeneralSettings::GeneralSettings(storm::settings::SettingsManager& settingsManager) : ModuleSettings(settingsManager, moduleName) {
                this->addOption(storm::settings::OptionBuilder(moduleName, helpOptionName, false, "Shows all available options, arguments and descriptions.").setShortName(helpOptionShortName)
                                .addArgument(storm::settings::ArgumentBuilder::createStringArgument("hint", "A regular expression to show help for all matching entities or 'all' for the complete help.").setDefaultValueString("all").build()).build());
                this->addOption(storm::settings::OptionBuilder(moduleName, versionOptionName, false, "Prints the version information.").build());
                this->addOption(storm::settings::OptionBuilder(moduleName, verboseOptionName, false, "Enables more verbose output.").setShortName(verboseOptionShortName).build());
                this->addOption(storm::settings::OptionBuilder(moduleName, precisionOptionName, false, "The internally used precision.").setShortName(precisionOptionShortName)
                                .addArgument(storm::settings::ArgumentBuilder::createDoubleArgument("value", "The precision to use.").setDefaultValueDouble(1e-06).addValidationFunctionDouble(storm::settings::ArgumentValidators::doubleRangeValidatorExcluding(0.0, 1.0)).build()).build());
                this->addOption(storm::settings::OptionBuilder(moduleName, exportDotOptionName, "", "If given, the loaded model will be written to the specified file in the dot format.")
                                .addArgument(storm::settings::ArgumentBuilder::createStringArgument("filename", "The name of the file to which the model is to be written.").build()).build());
                this->addOption(storm::settings::OptionBuilder(moduleName, configOptionName, false, "If given, this file will be read and parsed for additional configuration settings.").setShortName(configOptionShortName)
                                .addArgument(storm::settings::ArgumentBuilder::createStringArgument("filename", "The name of the file from which to read the configuration.").addValidationFunctionString(storm::settings::ArgumentValidators::existingReadableFileValidator()).build()).build());
                this->addOption(storm::settings::OptionBuilder(moduleName, explicitOptionName, false, "Parses the model given in an explicit (sparse) representation.").setShortName(explicitOptionShortName)
                                .addArgument(storm::settings::ArgumentBuilder::createStringArgument("transition filename", "The name of the file from which to read the transitions.").addValidationFunctionString(storm::settings::ArgumentValidators::existingReadableFileValidator()).build())
                                .addArgument(storm::settings::ArgumentBuilder::createStringArgument("labeling filename", "The name of the file from which to read the state labeling.").addValidationFunctionString(storm::settings::ArgumentValidators::existingReadableFileValidator()).build()).build());
                this->addOption(storm::settings::OptionBuilder(moduleName, symbolicOptionName, false, "Parses the model given in a symbolic representation.").setShortName(symbolicOptionShortName)
                                .addArgument(storm::settings::ArgumentBuilder::createStringArgument("filename", "The name of the file from which to read the symbolic model.").addValidationFunctionString(storm::settings::ArgumentValidators::existingReadableFileValidator()).build()).build());
                this->addOption(storm::settings::OptionBuilder(moduleName, pctlOptionName, false, "Specifies a PCTL formula that is to be checked on the model.")
                                .addArgument(storm::settings::ArgumentBuilder::createStringArgument("formula", "The formula to check.").build()).build());
                this->addOption(storm::settings::OptionBuilder(moduleName, pctlFileOptionName, false, "Specifies the PCTL formulas that are to be checked on the model.")
                                .addArgument(storm::settings::ArgumentBuilder::createStringArgument("filename", "The file from which to read the PCTL formulas.").addValidationFunctionString(storm::settings::ArgumentValidators::existingReadableFileValidator()).build()).build());
                this->addOption(storm::settings::OptionBuilder(moduleName, cslOptionName, false, "Specifies a CSL formula that is to be checked on the model.")
                                .addArgument(storm::settings::ArgumentBuilder::createStringArgument("formula", "The formula to check.").build()).build());
                this->addOption(storm::settings::OptionBuilder(moduleName, cslFileOptionName, false, "Specifies the CSL formulas that are to be checked on the model.")
                                .addArgument(storm::settings::ArgumentBuilder::createStringArgument("filename", "The file from which to read the CSL formulas.").addValidationFunctionString(storm::settings::ArgumentValidators::existingReadableFileValidator()).build()).build());
                this->addOption(storm::settings::OptionBuilder(moduleName, ltlOptionName, false, "Specifies an LTL formula that is to be checked on the model.")
                                .addArgument(storm::settings::ArgumentBuilder::createStringArgument("formula", "The formula to check.").build()).build());
                this->addOption(storm::settings::OptionBuilder(moduleName, ltlFileOptionName, false, "Specifies the LTL formulas that are to be checked on the model.")
                                .addArgument(storm::settings::ArgumentBuilder::createStringArgument("filename", "The file from which to read the LTL formulas.").addValidationFunctionString(storm::settings::ArgumentValidators::existingReadableFileValidator()).build()).build());
                this->addOption(storm::settings::OptionBuilder(moduleName, counterexampleOptionName, false, "Generates a counterexample for the given PRCTL formulas if not satisfied by the model")
                                .addArgument(storm::settings::ArgumentBuilder::createStringArgument("filename", "The name of the file to which the counterexample is to be written.").setDefaultValueString("-").setIsOptional(true).build()).setShortName(counterexampleOptionShortName).build());
                
                this->addOption(storm::settings::OptionBuilder(moduleName, transitionRewardsOptionName, "", "If given, the transition rewards are read from this file and added to the explicit model. Note that this requires the model to be given as an explicit model (i.e., via --" + explicitOptionName + ").")
                                .addArgument(storm::settings::ArgumentBuilder::createStringArgument("filename", "The file from which to read the transition rewards.").addValidationFunctionString(storm::settings::ArgumentValidators::existingReadableFileValidator()).build()).build());
                this->addOption(storm::settings::OptionBuilder(moduleName, stateRewardsOptionName, false, "If given, the state rewards are read from this file and added to the explicit model. Note that this requires the model to be given as an explicit model (i.e., via --" + explicitOptionName + ").")
                                .addArgument(storm::settings::ArgumentBuilder::createStringArgument("filename", "The file from which to read the state rewards.").addValidationFunctionString(storm::settings::ArgumentValidators::existingReadableFileValidator()).build()).build());
                this->addOption(storm::settings::OptionBuilder(moduleName, dontFixDeadlockOptionName, false, "If the model contains deadlock states, they need to be fixed by setting this option.").setShortName(dontFixDeadlockOptionShortName).build());
                
                std::vector<std::string> linearEquationSolver = {"gmm++", "native"};
                this->addOption(storm::settings::OptionBuilder(moduleName, eqSolverOptionName, false, "Sets which solver is preferred for solving systems of linear equations.")
                                .addArgument(storm::settings::ArgumentBuilder::createStringArgument("name", "The name of the solver to prefer. Available are: gmm++ and native.").addValidationFunctionString(storm::settings::ArgumentValidators::stringInListValidator(linearEquationSolver)).setDefaultValueString("gmm++").build()).build());
                this->addOption(storm::settings::OptionBuilder(moduleName, timeoutOptionName, false, "If given, computation will abort after the timeout has been reached.").setShortName(timeoutOptionShortName)
                                .addArgument(storm::settings::ArgumentBuilder::createUnsignedIntegerArgument("time", "The number of seconds after which to timeout.").setDefaultValueUnsignedInteger(0).build()).build());
                
                std::vector<std::string> lpSolvers = {"gurobi", "glpk"};
                this->addOption(storm::settings::OptionBuilder(moduleName, lpSolverOptionName, false, "Sets which LP solver is preferred.")
                                .addArgument(storm::settings::ArgumentBuilder::createStringArgument("name", "The name of an LP solver. Available are: gurobi and glpk.").addValidationFunctionString(storm::settings::ArgumentValidators::stringInListValidator(lpSolvers)).setDefaultValueString("glpk").build()).build());
                this->addOption(storm::settings::OptionBuilder(moduleName, constantsOptionName, false, "Specifies the constant replacements to use in symbolic models. Note that Note that this requires the model to be given as an symbolic model (i.e., via --" + symbolicOptionName + ").").setShortName(constantsOptionShortName)
                                .addArgument(storm::settings::ArgumentBuilder::createStringArgument("values", "A comma separated list of constants and their value, e.g. a=1,b=2,c=3.").setDefaultValueString("").build()).build());
                this->addOption(storm::settings::OptionBuilder(moduleName, statisticsOptionName, false, "Sets whether to display statistics if available.").setShortName(statisticsOptionShortName).build());

            }
            
            bool GeneralSettings::isHelpSet() const {
                return this->getOption(helpOptionName).getHasOptionBeenSet();
            }
            
            bool GeneralSettings::isVersionSet() const {
                return this->getOption(versionOptionName).getHasOptionBeenSet();
            }
            
            std::string GeneralSettings::getHelpModuleName() const {
                return this->getOption(helpOptionName).getArgumentByName("hint").getValueAsString();
            }
            
            bool GeneralSettings::isVerboseSet() const {
                return this->getOption(verboseOptionName).getHasOptionBeenSet();
            }
            
            double GeneralSettings::getPrecision() const {
                double value = this->getOption(precisionOptionName).getArgumentByName("value").getValueAsDouble();
                return value;
            }
            
            bool GeneralSettings::isExportDotSet() const {
                return this->getOption(exportDotOptionName).getHasOptionBeenSet();
            }
            
            std::string GeneralSettings::getExportDotFilename() const {
                return this->getOption(exportDotOptionName).getArgumentByName("filename").getValueAsString();
            }
            
            bool GeneralSettings::isConfigSet() const {
                return this->getOption(configOptionName).getHasOptionBeenSet();
            }
            
            std::string GeneralSettings::getConfigFilename() const {
                return this->getOption(configOptionName).getArgumentByName("filename").getValueAsString();
            }
            
            bool GeneralSettings::isExplicitSet() const {
                return this->getOption(explicitOptionName).getHasOptionBeenSet();
            }
            
            std::string GeneralSettings::getTransitionFilename() const {
                return this->getOption(explicitOptionName).getArgumentByName("transition filename").getValueAsString();
            }
                        
            std::string GeneralSettings::getLabelingFilename() const {
                return this->getOption(explicitOptionName).getArgumentByName("labeling filename").getValueAsString();
            }
            
            bool GeneralSettings::isSymbolicSet() const {
                return this->getOption(symbolicOptionName).getHasOptionBeenSet();
            }
            
            std::string GeneralSettings::getSymbolicModelFilename() const {
                return this->getOption(symbolicOptionName).getArgumentByName("filename").getValueAsString();
            }
            
            bool GeneralSettings::isPctlPropertySet() const {
                return this->getOption(pctlOptionName).getHasOptionBeenSet();
            }
            
            std::string GeneralSettings::getPctlProperty() const {
                return this->getOption(pctlOptionName).getArgumentByName("formula").getValueAsString();
            }
            
            bool GeneralSettings::isPctlFileSet() const {
                return this->getOption(pctlFileOptionName).getHasOptionBeenSet();
            }
            
            std::string GeneralSettings::getPctlPropertiesFilename() const {
                return this->getOption(pctlFileOptionName).getArgumentByName("filename").getValueAsString();
            }
            
            bool GeneralSettings::isCslPropertySet() const {
                return this->getOption(cslOptionName).getHasOptionBeenSet();
            }
            
            std::string GeneralSettings::getCslProperty() const {
                return this->getOption(cslOptionName).getArgumentByName("formula").getValueAsString();
            }
            
            bool GeneralSettings::isCslFileSet() const {
                return this->getOption(cslFileOptionName).getHasOptionBeenSet();
            }
            
            std::string GeneralSettings::getCslPropertiesFilename() const {
                return this->getOption(cslOptionName).getArgumentByName("filename").getValueAsString();
            }
            
            bool GeneralSettings::isLtlPropertySet() const {
                return this->getOption(ltlOptionName).getHasOptionBeenSet();
            }
            
            std::string GeneralSettings::getLtlProperty() const {
                return this->getOption(ltlOptionName).getArgumentByName("formula").getValueAsString();
            }
            
            bool GeneralSettings::isLtlFileSet() const {
                return this->getOption(ltlFileOptionName).getHasOptionBeenSet();
            }
            
            std::string GeneralSettings::getLtlPropertiesFilename() const {
                return this->getOption(ltlOptionName).getArgumentByName("filename").getValueAsString();
            }
            
            bool GeneralSettings::isTransitionRewardsSet() const {
                return this->getOption(transitionRewardsOptionName).getHasOptionBeenSet();
            }
            
            std::string GeneralSettings::getTransitionRewardsFilename() const {
                return this->getOption(transitionRewardsOptionName).getArgumentByName("filename").getValueAsString();
            }
            
            bool GeneralSettings::isStateRewardsSet() const {
                return this->getOption(stateRewardsOptionName).getHasOptionBeenSet();
            }
            
            std::string GeneralSettings::getStateRewardsFilename() const {
                return this->getOption(stateRewardsOptionName).getArgumentByName("filename").getValueAsString();
            }
            
            bool GeneralSettings::isCounterexampleSet() const {
                return this->getOption(counterexampleOptionName).getHasOptionBeenSet();
            }
            
            std::string GeneralSettings::getCounterexampleFilename() const {
                return this->getOption(counterexampleOptionName).getArgumentByName("filename").getValueAsString();
            }
            
            bool GeneralSettings::isDontFixDeadlocksSet() const {
                return this->getOption(dontFixDeadlockOptionName).getHasOptionBeenSet();
            }
            
            std::unique_ptr<storm::settings::SettingMemento> GeneralSettings::overrideDontFixDeadlocksSet(bool stateToSet) {
                return this->overrideOption(dontFixDeadlockOptionName, stateToSet);
            }
            
            bool GeneralSettings::isTimeoutSet() const {
                return this->getOption(timeoutOptionName).getHasOptionBeenSet();
            }
            
            uint_fast64_t GeneralSettings::getTimeoutInSeconds() const {
                return this->getOption(timeoutOptionName).getArgumentByName("time").getValueAsUnsignedInteger();
            }
            
            GeneralSettings::EquationSolver GeneralSettings::getEquationSolver() const {
                std::string equationSolverName = this->getOption(eqSolverOptionName).getArgumentByName("name").getValueAsString();
                if (equationSolverName == "gmm++") {
                    return GeneralSettings::EquationSolver::Gmmxx;
                } else if (equationSolverName == "native") {
                    return GeneralSettings::EquationSolver::Native;
                }
                STORM_LOG_THROW(false, storm::exceptions::IllegalArgumentValueException, "Unknown equation solver '" << equationSolverName << "'.");
            }
            
            GeneralSettings::LpSolver GeneralSettings::getLpSolver() const {
                std::string lpSolverName = this->getOption(lpSolverOptionName).getArgumentByName("name").getValueAsString();
                if (lpSolverName == "gurobi") {
                    return GeneralSettings::LpSolver::Gurobi;
                } else if (lpSolverName == "glpk") {
                    return GeneralSettings::LpSolver::glpk;
                }
                STORM_LOG_THROW(false, storm::exceptions::IllegalArgumentValueException, "Unknown LP solver '" << lpSolverName << "'.");
            }
            
            bool GeneralSettings::isConstantsSet() const {
                return this->getOption(constantsOptionName).getHasOptionBeenSet();
            }
            
            std::string GeneralSettings::getConstantDefinitionString() const {
                return this->getOption(constantsOptionName).getArgumentByName("values").getValueAsString();
            }
            
            bool GeneralSettings::isShowStatisticsSet() const {
                return this->getOption(statisticsOptionName).getHasOptionBeenSet();
            }
            
            bool GeneralSettings::check() const {
                // Ensure that the model was given either symbolically or explicitly.
                STORM_LOG_THROW(!isSymbolicSet() || !isExplicitSet(), storm::exceptions::InvalidSettingsException, "The model may be either given in an explicit or a symbolic format, but not both.");
                
                // Make sure that one "source" for properties is given.
                uint_fast64_t propertySources = 0 + (isPctlPropertySet() ? 1 : 0) + (isPctlFileSet() ? 1 : 0) + (isCslPropertySet() ? + 1 : 0) + (isCslFileSet() ? 1 : 0) + (isLtlPropertySet() ? 1 : 0) + (isLtlFileSet() ? 1 : 0);
                STORM_LOG_THROW(propertySources <= 1, storm::exceptions::InvalidSettingsException, "Please specify exactly one source of properties.");
                
                return true;
            }
            
        } // namespace modules
    } // namespace settings
} // namespace storm