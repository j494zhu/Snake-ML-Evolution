# Snake-ML-Evolution History

## Why this exists?

I recently finished my first machine learning project training a Multi-Layer Perceptron (MLP) to recognize handwritten digits (MNIST). It was fascinating, and I came into the idea of training something more interactive and dynamic: **Can a simple neeural network learn to survive?**

This repository documents my journey of building a einforcement Learning agent for the classic Snake game from scratch.

---

## The Environment (`game.py`)

To train an AI, a standard game loop for human players won't work, and I needed to decouple the game logic from the rendering speed.

I rebuilt the Snake game using `pygame`, with a specific **"AI Interface"**:

1.  **`play_step(action)`**: Instead of listening to keyboard events, the game waits for the agent to pass an `action` (e.g., `[1, 0, 0]` for "Go Straight").
2.  **Instant Feedback**: The function immediately returns:
    * `reward`: Did we do something good?
    * `done`: Did we die?
    * `score`: Current game score.
3.  **Visuals**: I kept the UI rendering enabled. This allows us to visually inspect the "personality" of the model as it trains—watching it go from suicidal to strategic is the best part.

*`helper.py` serves as our data analyst, plotting the score history in real-time so we can visualize the convergence.*

---

## The Brain (`agent.py` & `model.py`)

This is the core of the project. I used MLP to implement **Deep Q-Learning (DQN)** algorithm to train my AI. 

### 1. Feature Engineering: The 11 Parameters

I compressed the entire game state into a vector of just **11 integers**. My goal was to give the AI a **"Relative Perspective"** (First-person view) rather than an absolute map.

Every frame, the agent asks 11 simple Yes/No questions (0 or 1):

**A. Danger Perception (Relative to Head)**
* Is there danger **Straight Ahead**?
* Is there danger to my **Right**?
* Is there danger to my **Left**?
* *Note: These directions are relative to the direction of the snake. By using relative directions, the AI learns "Don't hit the wall on your right," regardless of whether it's facing North, South, East, or West.*, which allows my AI to learn faster. 

**B. Current Orientation**
* Am I moving **Left**?
* Am I moving **Right**?
* Am I moving **Up**?
* Am I moving **Down**?

**C. Target Location (The Food)**
* Is the food to my **Left**?
* Is the food to my **Right**?
* Is the food **Above** me?
* Is the food **Below** me?

### 2. The Model Architecture

Because the input is simplified to just 11 numbers, a lightweight MLP is sufficient to effectively approximate the Q-function without overfitting. As common practice, I applied ReLU to introduce non-linearity to prevent the network from degrading into a single linear regressor. 

* **Input Layer:** 11 Neurons
* **Hidden Layer:** 256 Neurons (ReLU activation)
* **Output Layer:** 3 Neurons (Action: `[Straight, Right, Left]`)

### 3. The Reward Mechanism

The key question is: How does the snake know it's doing well?
I designed the following mechanism: 
* **+10**: Eat Food (Positive Reinforcement).
* **-10**: Game Over (Hit wall or self).
* **0**: Move without eating (To prevent noise, but strict enough to force it to find food).

Under this mechanism, feedback is sparse—given only when the snake eats or dies. Consequently, the agent spends a significant amount of time continuously exploring randomly before stumbling upon food and establishing a positive feedback loop.

---

## Results

This 11 parameter MLP model demonstrated a clear learning curve, and is able to survive strtegically. 
![Training Plot](version_1_snake_ml_11_mlp_success/training_plot.png)

**Breakdown:**
* **The Exploration Phase (Games 0-90):**
    The agent spent the first ~90 games in a "warm-up" period. Scores hovered near zero as the snake wandered randomly, building up its experience memory (Replay Buffer) without yet understanding the connection between actions and rewards.

* **The Breakthrough (Game 100+):**
    A sharp inflection point occurs around Game 100. The model began to converge, showing a rapid and consistent increase in performance as it learned to associate the "food relative position" with the reward signal.

* **Final Stabilization:**
    By the end of 300 games, the **Mean Score** (orange line) reaches approximately **22**. This indicates the agent consistently finds food and survives. 

### Limitations:
* **High Variance (Score Instability):**
    While the mean score increases steadily, the standard deviation remains high. 

* **The "Dead End" Problem (U-Shape Traps):**
    The agent rarely dies from crashing into walls. Instead, most deaths occur due to **self-trapping**.
    * **Possible Reason:** The 11-parameter input vector only detects *immediate* danger (1 block away). The agent lacks the ability to see the global vision or larger neighboring areas, so it often enters U-shaped body formations (cul-de-sacs) to chase food, only to realize too late that it has trapped itself with no way out.