---
name: Autonomous Repo Refactor Agent
type: knowledge
version: 1.0.0
agent: CodeActAgent
triggers:
  - "take this repo"
  - "refactor repo"
  - "modify repo"
  - "edit repo to"
  - "transform repo"
  - "align repo with"
  - "repo vision"
---

# Autonomous Repo Refactor Agent

This microagent enables autonomous repository refactoring based on natural language vision descriptions. It can read entire repositories, understand architectural requirements, make comprehensive changes, test the results, and show live execution outcomes.

## Capabilities

- **Vision Understanding**: Parse natural language descriptions of desired architecture, patterns, or functionality
- **Code Analysis**: Read and understand existing codebases across multiple files and languages
- **Autonomous Refactoring**: Make comprehensive changes to align code with described vision
- **Testing & Debugging**: Automatically test changes and fix issues until functional
- **Live Execution**: Run the refactored code and show results in the OpenHands environment
- **Progress Tracking**: Provide real-time updates on refactoring progress

## Usage Pattern

When triggered, the agent follows this autonomous workflow:

1. **Repository Analysis**: Read and understand the current codebase structure
2. **Vision Parsing**: Extract specific requirements from natural language description
3. **Planning**: Create a comprehensive refactoring plan
4. **Implementation**: Make all necessary code changes
5. **Testing**: Run tests and validate functionality
6. **Debugging**: Fix any issues that arise
7. **Execution**: Run the final code and show results
8. **Reporting**: Provide summary of changes made

## Example Triggers

- "Take this repo, edit it to align with microservice architecture"
- "Refactor this Flask app to use FastAPI with async endpoints"
- "Transform this monolith into a modular plugin system"
- "Modify this repo to follow clean architecture principles"

## Technical Implementation

The agent uses:
- Advanced code parsing and AST analysis
- Pattern recognition for architectural transformations
- Automated testing frameworks
- Live code execution in sandboxed environment
- Real-time progress reporting through OpenHands UI

## Security Considerations

- All code execution happens in isolated environment
- Changes are tracked and can be reverted
- No external network access during refactoring unless explicitly required
- Sensitive data handling follows security best practices