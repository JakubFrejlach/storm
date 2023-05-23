// authors: Roman Andriushchenko, Jakub Frejlach

#ifndef STORM_SYNTHESIS_COUNTEREXAMPLEMDP_H
#define STORM_SYNTHESIS_COUNTEREXAMPLEMDP_H

#include "storm/storage/jani/Model.h"
#include "storm/logic/Formula.h"
#include "storm/models/sparse/Mdp.h"
#include "storm/modelchecker/results/ExplicitQuantitativeCheckResult.h"
#include "storm/modelchecker/results/ExplicitQualitativeCheckResult.h"
#include "storm/transformer/SubsystemBuilder.h"

#include "storm/models/sparse/Dtmc.h"
#include "storm/utility/Stopwatch.h"

namespace storm {
    namespace synthesis {

        template<typename ValueType = double, typename StateType = uint_fast64_t>
        class CounterexampleGeneratorMdp {

        using StormRow = std::vector<std::pair<StateType, ValueType>>;

        public:

            /*!
             * Preprocess the quotient MDP and its bound on the reachability
             * probability before constructing counterexamples from various
             * deterministic sub-MDPs (DTMCs).
             * @param quotient_mdp The quotient MDP.
             * @param hole_count Total number of holes.
             * @param mdp_holes For each state of a quotient MDP, a set of
             *   indices of significant holes.
             * @param formulae Formulae to check, can be both
             *   probabilistic and reward-based.
             */
            CounterexampleGeneratorMdp(
                storm::models::sparse::Mdp<ValueType> const& quotient_mdp,
                uint_fast64_t hole_count,
                std::vector<std::set<uint_fast64_t>> const& quotient_holes,
                std::vector<std::shared_ptr<storm::logic::Formula const>> const& formulae
                );

            /*!
             * Preprocess the MDP for the state space exploration
             * @param mdp explored MDP
             * @param state_map subMDP-MDP state mapping.
             * @param simple_holes bit vector indicating simple holes
             * @param assignment assignment of the holes
             * @param hole_position_generalization bool indicator if this variant is enabled
             */
            void prepareMdp(
                storm::models::sparse::Mdp<ValueType> const& Mdp,
                std::vector<uint_fast64_t> const& state_map,
                storm::storage::BitVector simple_holes,
                std::vector<uint_fast64_t> assignment,
                bool hole_position_generalization
                );

            /*!
             * - explore the reachable state space by one wave
             * - during each wave, expand only 'non-blocking' states (states with registered holes)
             * - if no non-blocking state remains, pick a blocking candidate with the least amount of unregistered holes
             * - register all holes in this blocking candidate, thus unblocking this state (and possibly many others)
             * @return boolean indicating whether the whole state space is explored
             */
            bool exploreWave ();

            /*!
             * Construct a counterexample to a prepared MDP and a formula with
             * the given index.
             * @param formula_index Formula index.
             * @param formula_bound Formula threshold for CE construction.
             * @param mdp_bounds MDP model checking result in the primary direction (NULL if not used).
             * @param mdp_quotient_state_mdp A mapping of MDP states to the states of a quotient MDP.
             * @return A list of holes relevant in the CE.
             */
            std::vector<uint_fast64_t> constructConflict(
                uint_fast64_t formula_index,
                ValueType formula_bound,
                std::shared_ptr<storm::modelchecker::ExplicitQuantitativeCheckResult<ValueType> const> mdp_bounds,
                std::vector<StateType> const& mdp_quotient_state_map
                );

        protected:

            /** Identify states of an MDP having some label. */
            std::shared_ptr<storm::modelchecker::ExplicitQualitativeCheckResult> labelStates(
                storm::models::sparse::Mdp<ValueType> const& mdp,
                storm::logic::Formula const& label
                );

            /**
             * Prepare data structures for sub-MDP construction.
             * @param formula_index Formula index.
             * @param mdp_bounds MDP model checking result in the primary direction.
             * @param matrix_dtmc (output) Copy of the transition matrix of the DTMC.
             * @param matrix_subdtmc (output) Matrix of shortcuts.
             * @param labeling_subdtdmc (output) Labeling marking target states.
             * @param reward_model_subdtmc (output) If the reward property is
             *   investigated, this map will contain exactly one reward model
             *   for the initial sub-DTMC.
             */
            void prepareSubmdp(
                uint_fast64_t formula_index,
                std::shared_ptr<storm::modelchecker::ExplicitQuantitativeCheckResult<ValueType> const> mdp_bounds,
                std::vector<StateType> const& mdp_quotient_state_map
                );

            /**
             * Expand new wave and model check resulting rerouting of a MDP.
             * @param formula_index Formula index.
             * @param formula_bound Formula bound.
             * @param mdp_bounds MDP model checking result in the primary direction.
             * @param mdp_quotient_state_mdp A mapping of MDP states to the states of a quotient MDP.
             * @return first: true if the rerouting still satisfies the formula, second: true if this is the last wave
             */
            std::pair<bool,bool> expandAndCheck(
                uint_fast64_t formula_index,
                ValueType formula_bound,
                std::shared_ptr<storm::modelchecker::ExplicitQuantitativeCheckResult<ValueType> const> mdp_bounds,
                std::vector<StateType> const& mdp_quotient_state_map
                );

            // Quotient MDP
            storm::models::sparse::Mdp<ValueType> const& quotient_mdp;
            // Number of significant holes
            uint_fast64_t hole_count;
            // Significant holes in Quotient states
            std::vector<std::set<uint_fast64_t>> quotient_holes;
            // Simple holes bit vector
            storm::storage::BitVector simple_holes;
            // Generalized holes
            storm::storage::BitVector generalized_holes;
            // Holes assigment
            std::vector<uint_fast64_t> assignment;

            // Formula bounds: safety (<,<=) or liveness (>,>=)
            std::vector<bool> formula_safety;
            // Formula types: probability (false) or reward-based (true)
            std::vector<bool> formula_reward;
            // Reward model names for reward formulae
            std::vector<std::string> formula_reward_name;

            // Until label for sub-mdps
            const std::string until_label = "__until__";
            // Target label for sub-mdps
            const std::string target_label = "__target__";
            // Modified operator formulae to apply to sub-mdps: P~?["__until" U "__target__"] or P~?[F "__target__"]
            std::vector<std::shared_ptr<storm::logic::Formula>> formula_modified;
            // Flags for until states
            std::vector<std::shared_ptr<storm::modelchecker::ExplicitQualitativeCheckResult const>> mdp_untils;
            // Flags for target states
            std::vector<std::shared_ptr<storm::modelchecker::ExplicitQualitativeCheckResult const>> mdp_targets;

            // MDP under investigation
            std::shared_ptr<storm::models::sparse::Mdp<ValueType>> mdp;
            // std::shared_ptr<storm::models::sparse::Dtmc<ValueType>> dtmc;
            // MDP to quotient MDP state mappingtransition matrix of the DTMC
            std::vector<uint_fast64_t> state_map;
            // For each hole, a wave when it was registered (0 = unregistered).
            std::vector<uint_fast64_t> hole_wave;
            // For each wave, a set of states that were expanded.
            std::vector<std::vector<StateType>> wave_states;
            // non-blocking horizon
            std::stack<StateType> state_horizon;
            // horizon containing, for a current wave, only blocking states
            std::vector<StateType> state_horizon_blocking;
            // relevant holes
            std::vector<std::set<uint_fast64_t>> mdp_holes;
            // Labeling marking target states
            // relevant holes count
            std::vector<uint_fast64_t> unregistered_holes_count;
            // true if the state was reached during exploration (expanded states + both horizons)
            storm::storage::BitVector reachable_flag;
            // blocking state containing currently the least number of unregistered holes + flag if the value was set
            StateType blocking_candidate;
            bool blocking_candidate_set;
            // wave increases by one when new holes of a blocking candidate are registered
            uint_fast64_t current_wave;
            // indicator that the hole was generalized
            bool hole_generalized;
            storm::storage::BitVector actions_to_keep;
            // transition matrix of the MDP
            std::vector<std::vector<StormRow>> matrix_submdp;
            // Labeling marking target states
            storm::models::sparse::StateLabeling labeling_submdp;
            // If the reward property is investigated, this map will contain exactly one reward model for the initial sub-DTMC
            std::unordered_map<std::string, storm::models::sparse::StandardRewardModel<ValueType>> reward_models_submdp;
            // Hint for future model checking.
            std::unique_ptr<storm::modelchecker::CheckResult> hint_result;
            // flag if the hole position generalization is enabled
            bool hole_position_generalization;

            // Profiling
            storm::utility::Stopwatch timer_conflict;
            storm::utility::Stopwatch timer_model_check;

        };

    } // namespace research
} // namespace storm

#endif  /* STORM_SYNTHESIS_COUNTEREXAMPLEMDP_H */
