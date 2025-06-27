#!/usr/bin/env python3
"""
WADE Self-Evolution Engine - Autonomous learning and improvement
Scores, retains, and mutates successful task chains
"""

import json
import random
import hashlib
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import copy

@dataclass
class EvolutionScore:
    chain_id: str
    execution_success: float  # 0.0 - 1.0
    stealth_score: float     # 0.0 - 1.0 (lower detection)
    efficiency_score: float  # 0.0 - 1.0 (speed/resource usage)
    innovation_score: float  # 0.0 - 1.0 (novel approaches)
    safety_score: float      # 0.0 - 1.0 (safety compliance)
    overall_score: float     # Weighted combination
    execution_count: int
    last_executed: datetime
    mutation_generation: int = 0
    parent_chain_id: Optional[str] = None

@dataclass
class ChainMutation:
    mutation_id: str
    parent_chain_id: str
    mutation_type: str  # "parameter", "structure", "agent", "hybrid"
    changes: List[Dict[str, Any]]
    success_prediction: float
    created_at: datetime

class EvolutionEngine:
    def __init__(self, workspace_path: str = "/workspace/wade_env"):
        self.workspace_path = Path(workspace_path)
        self.memory_path = self.workspace_path / "memory"
        self.evolution_path = self.memory_path / "evolution"
        self.chains_path = self.memory_path / "chains"
        
        # Create directories
        self.evolution_path.mkdir(parents=True, exist_ok=True)
        self.chains_path.mkdir(parents=True, exist_ok=True)
        
        # Evolution parameters
        self.mutation_rate = 0.3
        self.crossover_rate = 0.7
        self.elite_retention = 0.2
        self.max_generations = 50
        self.population_size = 20
        
        # Scoring weights
        self.score_weights = {
            "execution_success": 0.35,
            "stealth_score": 0.20,
            "efficiency_score": 0.15,
            "innovation_score": 0.15,
            "safety_score": 0.15
        }
        
        # Load existing evolution data
        self.evolution_scores: Dict[str, EvolutionScore] = self._load_evolution_data()
        self.successful_patterns: List[Dict] = self._load_successful_patterns()
    
    def score_chain_execution(self, chain_result: Dict[str, Any], execution_metrics: Dict[str, Any]) -> EvolutionScore:
        """Score a completed task chain execution"""
        chain_id = chain_result["chain_id"]
        
        # Calculate individual scores
        execution_success = self._calculate_execution_success(chain_result)
        stealth_score = self._calculate_stealth_score(execution_metrics)
        efficiency_score = self._calculate_efficiency_score(execution_metrics)
        innovation_score = self._calculate_innovation_score(chain_result)
        safety_score = self._calculate_safety_score(chain_result, execution_metrics)
        
        # Calculate weighted overall score
        overall_score = (
            execution_success * self.score_weights["execution_success"] +
            stealth_score * self.score_weights["stealth_score"] +
            efficiency_score * self.score_weights["efficiency_score"] +
            innovation_score * self.score_weights["innovation_score"] +
            safety_score * self.score_weights["safety_score"]
        )
        
        # Update or create evolution score
        if chain_id in self.evolution_scores:
            existing = self.evolution_scores[chain_id]
            existing.execution_count += 1
            existing.last_executed = datetime.now()
            # Update scores with exponential moving average
            alpha = 0.3
            existing.execution_success = alpha * execution_success + (1 - alpha) * existing.execution_success
            existing.stealth_score = alpha * stealth_score + (1 - alpha) * existing.stealth_score
            existing.efficiency_score = alpha * efficiency_score + (1 - alpha) * existing.efficiency_score
            existing.innovation_score = alpha * innovation_score + (1 - alpha) * existing.innovation_score
            existing.safety_score = alpha * safety_score + (1 - alpha) * existing.safety_score
            existing.overall_score = alpha * overall_score + (1 - alpha) * existing.overall_score
            score = existing
        else:
            score = EvolutionScore(
                chain_id=chain_id,
                execution_success=execution_success,
                stealth_score=stealth_score,
                efficiency_score=efficiency_score,
                innovation_score=innovation_score,
                safety_score=safety_score,
                overall_score=overall_score,
                execution_count=1,
                last_executed=datetime.now()
            )
            self.evolution_scores[chain_id] = score
        
        # Save updated scores
        self._save_evolution_data()
        
        # Extract successful patterns if score is high
        if overall_score > 0.8:
            self._extract_successful_pattern(chain_result, score)
        
        return score
    
    def _calculate_execution_success(self, chain_result: Dict[str, Any]) -> float:
        """Calculate execution success score"""
        if "success_score" in chain_result:
            return float(chain_result["success_score"])
        
        # Fallback calculation
        total_tasks = len(chain_result.get("tasks", []))
        if total_tasks == 0:
            return 0.0
        
        completed_tasks = sum(1 for task in chain_result["tasks"] 
                            if task.get("status") == "completed")
        return completed_tasks / total_tasks
    
    def _calculate_stealth_score(self, execution_metrics: Dict[str, Any]) -> float:
        """Calculate stealth/detection avoidance score"""
        # Higher score = better stealth (lower detection)
        detection_events = execution_metrics.get("detection_events", 0)
        network_noise = execution_metrics.get("network_requests", 0)
        file_modifications = execution_metrics.get("file_modifications", 0)
        
        # Normalize and invert (lower activity = higher stealth)
        stealth = 1.0 - min(1.0, (detection_events * 0.5 + network_noise * 0.01 + file_modifications * 0.02))
        return max(0.0, stealth)
    
    def _calculate_efficiency_score(self, execution_metrics: Dict[str, Any]) -> float:
        """Calculate efficiency score based on resource usage"""
        execution_time = execution_metrics.get("execution_time_seconds", 300)
        memory_usage = execution_metrics.get("peak_memory_mb", 100)
        cpu_usage = execution_metrics.get("avg_cpu_percent", 50)
        
        # Normalize scores (lower usage = higher efficiency)
        time_score = max(0.0, 1.0 - (execution_time / 600))  # 10 min baseline
        memory_score = max(0.0, 1.0 - (memory_usage / 500))  # 500MB baseline
        cpu_score = max(0.0, 1.0 - (cpu_usage / 100))
        
        return (time_score + memory_score + cpu_score) / 3
    
    def _calculate_innovation_score(self, chain_result: Dict[str, Any]) -> float:
        """Calculate innovation score based on novel approaches"""
        chain_hash = self._get_chain_structure_hash(chain_result)
        
        # Check similarity to existing patterns
        similarity_scores = []
        for pattern in self.successful_patterns:
            pattern_hash = pattern.get("structure_hash", "")
            similarity = self._calculate_hash_similarity(chain_hash, pattern_hash)
            similarity_scores.append(similarity)
        
        if not similarity_scores:
            return 1.0  # First chain is maximally innovative
        
        # Innovation = 1 - max_similarity
        max_similarity = max(similarity_scores)
        return max(0.0, 1.0 - max_similarity)
    
    def _calculate_safety_score(self, chain_result: Dict[str, Any], execution_metrics: Dict[str, Any]) -> float:
        """Calculate safety compliance score"""
        execution_mode = chain_result.get("execution_mode", "simulation")
        
        # Base safety score
        if execution_mode == "simulation":
            base_score = 1.0
        else:
            base_score = 0.7  # Live mode inherently less safe
        
        # Deduct for risky operations
        risk_factors = execution_metrics.get("risk_factors", [])
        risk_penalty = len(risk_factors) * 0.1
        
        # Deduct for errors/exceptions
        error_count = execution_metrics.get("error_count", 0)
        error_penalty = min(0.3, error_count * 0.05)
        
        safety_score = base_score - risk_penalty - error_penalty
        return max(0.0, safety_score)
    
    def generate_mutations(self, parent_chain_id: str, num_mutations: int = 5) -> List[Dict[str, Any]]:
        """Generate mutations of a successful chain"""
        if parent_chain_id not in self.evolution_scores:
            raise ValueError(f"Parent chain {parent_chain_id} not found in evolution data")
        
        parent_score = self.evolution_scores[parent_chain_id]
        parent_chain = self._load_chain_config(parent_chain_id)
        
        mutations = []
        for i in range(num_mutations):
            mutation = self._create_mutation(parent_chain, parent_score)
            mutations.append(mutation)
        
        return mutations
    
    def _create_mutation(self, parent_chain: Dict[str, Any], parent_score: EvolutionScore) -> Dict[str, Any]:
        """Create a single mutation of a parent chain"""
        mutation_type = random.choice(["parameter", "structure", "agent", "hybrid"])
        mutated_chain = copy.deepcopy(parent_chain)
        changes = []
        
        if mutation_type == "parameter":
            changes = self._mutate_parameters(mutated_chain)
        elif mutation_type == "structure":
            changes = self._mutate_structure(mutated_chain)
        elif mutation_type == "agent":
            changes = self._mutate_agents(mutated_chain)
        elif mutation_type == "hybrid":
            changes = self._mutate_hybrid(mutated_chain)
        
        # Generate new chain ID
        mutated_chain["chain_id"] = self._generate_chain_id(mutated_chain)
        mutated_chain["parent_chain_id"] = parent_chain["chain_id"]
        mutated_chain["mutation_generation"] = parent_score.mutation_generation + 1
        
        # Predict success based on parent score and mutation type
        success_prediction = self._predict_mutation_success(parent_score, mutation_type, changes)
        
        # Create mutation record
        mutation = ChainMutation(
            mutation_id=mutated_chain["chain_id"],
            parent_chain_id=parent_chain["chain_id"],
            mutation_type=mutation_type,
            changes=changes,
            success_prediction=success_prediction,
            created_at=datetime.now()
        )
        
        # Save mutation
        self._save_chain_config(mutated_chain)
        self._save_mutation_record(mutation)
        
        return mutated_chain
    
    def _mutate_parameters(self, chain: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Mutate task parameters"""
        changes = []
        
        for task in chain.get("tasks", []):
            if random.random() < self.mutation_rate:
                # Mutate timeout
                if "timeout" in task:
                    old_timeout = task["timeout"]
                    task["timeout"] = max(30, int(old_timeout * random.uniform(0.5, 2.0)))
                    changes.append({
                        "type": "parameter_change",
                        "task": task["name"],
                        "parameter": "timeout",
                        "old_value": old_timeout,
                        "new_value": task["timeout"]
                    })
                
                # Mutate priority
                if "priority" in task:
                    old_priority = task["priority"]
                    task["priority"] = random.randint(1, 5)
                    changes.append({
                        "type": "parameter_change",
                        "task": task["name"],
                        "parameter": "priority",
                        "old_value": old_priority,
                        "new_value": task["priority"]
                    })
        
        return changes
    
    def _mutate_structure(self, chain: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Mutate chain structure (add/remove/reorder tasks)"""
        changes = []
        tasks = chain.get("tasks", [])
        
        if random.random() < 0.3 and len(tasks) > 1:
            # Remove a non-critical task
            removable_tasks = [t for t in tasks if t.get("priority", 1) < 4]
            if removable_tasks:
                task_to_remove = random.choice(removable_tasks)
                tasks.remove(task_to_remove)
                changes.append({
                    "type": "task_removal",
                    "task": task_to_remove["name"]
                })
        
        if random.random() < 0.2:
            # Add a new task
            new_task = self._generate_random_task()
            tasks.append(new_task)
            changes.append({
                "type": "task_addition",
                "task": new_task["name"]
            })
        
        if random.random() < 0.4 and len(tasks) > 2:
            # Reorder tasks (while respecting dependencies)
            self._shuffle_compatible_tasks(tasks)
            changes.append({
                "type": "task_reordering",
                "new_order": [t["name"] for t in tasks]
            })
        
        return changes
    
    def _mutate_agents(self, chain: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Mutate agent assignments"""
        changes = []
        agent_options = ["Planner", "Coder", "Tester", "Recon", "Payload", "Analyzer"]
        
        for task in chain.get("tasks", []):
            if random.random() < self.mutation_rate:
                old_agent = task["agent"]
                # Choose a different agent
                available_agents = [a for a in agent_options if a != old_agent]
                task["agent"] = random.choice(available_agents)
                changes.append({
                    "type": "agent_change",
                    "task": task["name"],
                    "old_agent": old_agent,
                    "new_agent": task["agent"]
                })
        
        return changes
    
    def _mutate_hybrid(self, chain: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply multiple mutation types"""
        changes = []
        changes.extend(self._mutate_parameters(chain))
        if random.random() < 0.5:
            changes.extend(self._mutate_agents(chain))
        if random.random() < 0.3:
            changes.extend(self._mutate_structure(chain))
        return changes
    
    def get_top_performing_chains(self, limit: int = 10) -> List[Tuple[str, EvolutionScore]]:
        """Get top performing chains by overall score"""
        sorted_chains = sorted(
            self.evolution_scores.items(),
            key=lambda x: x[1].overall_score,
            reverse=True
        )
        return sorted_chains[:limit]
    
    def get_evolution_insights(self) -> Dict[str, Any]:
        """Get insights about evolution progress"""
        if not self.evolution_scores:
            return {"message": "No evolution data available"}
        
        scores = list(self.evolution_scores.values())
        
        insights = {
            "total_chains": len(scores),
            "avg_overall_score": np.mean([s.overall_score for s in scores]),
            "best_score": max(s.overall_score for s in scores),
            "worst_score": min(s.overall_score for s in scores),
            "total_executions": sum(s.execution_count for s in scores),
            "successful_patterns": len(self.successful_patterns),
            "score_distribution": {
                "excellent": len([s for s in scores if s.overall_score >= 0.9]),
                "good": len([s for s in scores if 0.7 <= s.overall_score < 0.9]),
                "average": len([s for s in scores if 0.5 <= s.overall_score < 0.7]),
                "poor": len([s for s in scores if s.overall_score < 0.5])
            },
            "top_performers": [
                {
                    "chain_id": chain_id,
                    "score": score.overall_score,
                    "executions": score.execution_count,
                    "generation": score.mutation_generation
                }
                for chain_id, score in self.get_top_performing_chains(5)
            ]
        }
        
        return insights
    
    def _extract_successful_pattern(self, chain_result: Dict[str, Any], score: EvolutionScore):
        """Extract and store successful patterns"""
        pattern = {
            "chain_id": chain_result["chain_id"],
            "structure_hash": self._get_chain_structure_hash(chain_result),
            "task_sequence": [task["agent"] for task in chain_result.get("tasks", [])],
            "success_factors": {
                "execution_success": score.execution_success,
                "stealth_score": score.stealth_score,
                "efficiency_score": score.efficiency_score,
                "innovation_score": score.innovation_score,
                "safety_score": score.safety_score
            },
            "extracted_at": datetime.now().isoformat(),
            "execution_count": score.execution_count
        }
        
        # Avoid duplicates
        existing_hashes = [p.get("structure_hash") for p in self.successful_patterns]
        if pattern["structure_hash"] not in existing_hashes:
            self.successful_patterns.append(pattern)
            self._save_successful_patterns()
    
    def _get_chain_structure_hash(self, chain: Dict[str, Any]) -> str:
        """Generate hash representing chain structure"""
        structure = {
            "task_count": len(chain.get("tasks", [])),
            "agent_sequence": [task["agent"] for task in chain.get("tasks", [])],
            "dependency_pattern": [task.get("depends_on", []) for task in chain.get("tasks", [])]
        }
        structure_str = json.dumps(structure, sort_keys=True)
        return hashlib.sha256(structure_str.encode()).hexdigest()[:16]
    
    def _calculate_hash_similarity(self, hash1: str, hash2: str) -> float:
        """Calculate similarity between two hashes"""
        if not hash1 or not hash2:
            return 0.0
        
        # Simple Hamming distance for hex strings
        if len(hash1) != len(hash2):
            return 0.0
        
        matches = sum(c1 == c2 for c1, c2 in zip(hash1, hash2))
        return matches / len(hash1)
    
    def _predict_mutation_success(self, parent_score: EvolutionScore, mutation_type: str, changes: List[Dict]) -> float:
        """Predict success probability of a mutation"""
        base_prediction = parent_score.overall_score
        
        # Adjust based on mutation type
        type_modifiers = {
            "parameter": 0.9,  # Conservative changes
            "structure": 0.7,  # More risky
            "agent": 0.8,      # Medium risk
            "hybrid": 0.6      # Highest risk
        }
        
        modifier = type_modifiers.get(mutation_type, 0.8)
        change_penalty = min(0.3, len(changes) * 0.05)  # More changes = more risk
        
        prediction = base_prediction * modifier - change_penalty
        return max(0.1, min(0.95, prediction))
    
    def _generate_random_task(self) -> Dict[str, Any]:
        """Generate a random task for mutations"""
        agents = ["Planner", "Coder", "Tester", "Analyzer"]
        task_names = [
            "Validate input", "Process data", "Generate output", 
            "Perform analysis", "Create report", "Execute validation"
        ]
        
        return {
            "name": random.choice(task_names),
            "agent": random.choice(agents),
            "depends_on": [],
            "output_var": f"output_{random.randint(1000, 9999)}",
            "input_vars": [],
            "priority": random.randint(1, 3),
            "timeout": random.choice([60, 120, 300, 600])
        }
    
    def _shuffle_compatible_tasks(self, tasks: List[Dict[str, Any]]):
        """Shuffle tasks while respecting dependencies"""
        # Simple implementation - just shuffle tasks with no dependencies
        independent_tasks = [t for t in tasks if not t.get("depends_on")]
        if len(independent_tasks) > 1:
            random.shuffle(independent_tasks)
    
    def _generate_chain_id(self, chain: Dict[str, Any]) -> str:
        """Generate unique chain ID"""
        chain_str = json.dumps(chain, sort_keys=True)
        return hashlib.sha256(chain_str.encode()).hexdigest()[:12]
    
    def _load_evolution_data(self) -> Dict[str, EvolutionScore]:
        """Load evolution scores from disk"""
        evolution_file = self.evolution_path / "scores.json"
        if not evolution_file.exists():
            return {}
        
        try:
            with open(evolution_file, 'r') as f:
                data = json.load(f)
            
            scores = {}
            for chain_id, score_data in data.items():
                score_data["last_executed"] = datetime.fromisoformat(score_data["last_executed"])
                scores[chain_id] = EvolutionScore(**score_data)
            
            return scores
        except Exception as e:
            print(f"Error loading evolution data: {e}")
            return {}
    
    def _save_evolution_data(self):
        """Save evolution scores to disk"""
        evolution_file = self.evolution_path / "scores.json"
        
        data = {}
        for chain_id, score in self.evolution_scores.items():
            score_dict = asdict(score)
            score_dict["last_executed"] = score.last_executed.isoformat()
            data[chain_id] = score_dict
        
        with open(evolution_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_successful_patterns(self) -> List[Dict]:
        """Load successful patterns from disk"""
        patterns_file = self.evolution_path / "patterns.json"
        if not patterns_file.exists():
            return []
        
        try:
            with open(patterns_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading patterns: {e}")
            return []
    
    def _save_successful_patterns(self):
        """Save successful patterns to disk"""
        patterns_file = self.evolution_path / "patterns.json"
        with open(patterns_file, 'w') as f:
            json.dump(self.successful_patterns, f, indent=2)
    
    def _load_chain_config(self, chain_id: str) -> Dict[str, Any]:
        """Load chain configuration from disk"""
        chain_file = self.chains_path / f"{chain_id}.json"
        if not chain_file.exists():
            raise FileNotFoundError(f"Chain config {chain_id} not found")
        
        with open(chain_file, 'r') as f:
            return json.load(f)
    
    def _save_chain_config(self, chain: Dict[str, Any]):
        """Save chain configuration to disk"""
        chain_id = chain["chain_id"]
        chain_file = self.chains_path / f"{chain_id}.json"
        with open(chain_file, 'w') as f:
            json.dump(chain, f, indent=2)
    
    def _save_mutation_record(self, mutation: ChainMutation):
        """Save mutation record to disk"""
        mutations_file = self.evolution_path / "mutations.jsonl"
        mutation_dict = asdict(mutation)
        mutation_dict["created_at"] = mutation.created_at.isoformat()
        
        with open(mutations_file, 'a') as f:
            f.write(json.dumps(mutation_dict) + '\n')

if __name__ == "__main__":
    # Test evolution engine
    engine = EvolutionEngine()
    
    # Simulate scoring a chain
    test_chain_result = {
        "chain_id": "test_chain_001",
        "name": "Test Chain",
        "success_score": 0.85,
        "execution_mode": "simulation",
        "tasks": [
            {"name": "Task 1", "agent": "Planner", "status": "completed"},
            {"name": "Task 2", "agent": "Coder", "status": "completed"},
            {"name": "Task 3", "agent": "Tester", "status": "failed"}
        ]
    }
    
    test_metrics = {
        "execution_time_seconds": 120,
        "peak_memory_mb": 150,
        "avg_cpu_percent": 30,
        "detection_events": 0,
        "network_requests": 5,
        "file_modifications": 3,
        "error_count": 1,
        "risk_factors": []
    }
    
    score = engine.score_chain_execution(test_chain_result, test_metrics)
    print("Evolution Score:", asdict(score))
    
    insights = engine.get_evolution_insights()
    print("\nEvolution Insights:")
    print(json.dumps(insights, indent=2))