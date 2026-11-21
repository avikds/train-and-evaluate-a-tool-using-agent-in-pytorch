"""
Train and Evaluate a Tool-Using Agent in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - run_tool
import random

ITEMS = ["apple", "pear", "fig", "plum", "kiwi", "lime", "mango", "peach", "grape", "melon",
         "lemon", "cherry", "olive", "date", "guava", "papaya", "quince", "banana", "orange", "apricot"]

# The warehouse the lookup tool reads: item -> units in stock.
# Every value is different.
STORE = dict(zip(ITEMS, [17, 5, 42, 8, 31, 26, 3, 49, 12, 38, 21, 9, 44, 15, 0, 36, 28, 7, 33, 23]))

# The same warehouse after a delivery: every stock has changed.
RESTOCKED = dict(zip(ITEMS, [30, 11, 2, 47, 19, 6, 40, 13, 35, 1, 27, 45, 16, 4, 22, 10, 48, 25, 14, 39]))

SPECIAL = ["<pad>", "<unk>", "<q>", "</q>", "<call>", "</call>", "<result>", "</result>", "<answer>", "</answer>"]

VOCAB = (SPECIAL + ["what", "is", "stock", "of", "plus", "+", "-", "*", "calc", "lookup", "error"]
         + ITEMS + [str(n) for n in range(100)])

STOI = {token: i for i, token in enumerate(VOCAB)}


def solve(question, store):
    """The correct answer (an int) to a question, computed directly from the store."""
    q = question.split()

    if q[0] == "what":
        a, b = int(q[2]), int(q[4])
        return a + b if q[3] == "+" else a - b

    if len(q) == 7:
        return store[q[2]] + store[q[6]]

    return store[q[2]] + int(q[4])


def random_store(seed):
    """A warehouse with the same items and random stocks (all different, 0 to 49)."""
    values = list(range(50))
    random.Random(seed).shuffle(values)
    return dict(zip(ITEMS, values))


def make_tasks(n, seed):
    """n different tasks, one third of each kind: 'calc', 'add' and 'total'."""
    rng = random.Random(seed)
    tasks, seen = [], set()

    while len(tasks) < n:
        kind = ("calc", "add", "total")[len(tasks) % 3]

        if kind == "calc":
            a, b, op = rng.randrange(50), rng.randrange(50), rng.choice("+-")

            if op == "-" and b > a:
                a, b = b, a

            question = f"what is {a} {op} {b}"

        elif kind == "add":
            question = f"stock of {rng.choice(ITEMS)} plus {rng.randrange(50)}"

        else:
            x = rng.choice(ITEMS)
            y = rng.choice([i for i in ITEMS if i != x])
            question = f"stock of {x} plus stock of {y}"

        if question not in seen:
            seen.add(question)
            tasks.append({
                "id": f"s{seed}-{len(tasks):03d}",
                "kind": kind,
                "question": question
            })

    return tasks


def teacher_policy(transcript):
    """A scripted expert. Reads the transcript so far and returns the next correct action."""
    toks = transcript.split()
    q = toks[1:toks.index("</q>")]

    results = [toks[i + 1] for i, t in enumerate(toks) if t == "<result>"]

    if q[0] == "what":
        plan = [["calc", q[2], q[3], q[4]]]

    elif len(q) == 7:
        plan = [
            ["lookup", q[2]],
            ["lookup", q[6]],
            ["calc", "$0", "+", "$1"]
        ]

    else:
        plan = [
            ["lookup", q[2]],
            ["calc", "$0", "+", q[4]]
        ]

    if len(results) < len(plan):
        step = [
            results[int(a[1:])] if a.startswith("$") else a
            for a in plan[len(results)]
        ]
        return "<call> " + " ".join(step) + " </call>"

    return "<answer> " + results[-1] + " </answer>"


def make_teacher(error_rate, seed):
    """The expert with mistakes: each action is replaced by a wrong one with probability error_rate."""
    rng = random.Random(seed)

    def policy(transcript):
        toks = teacher_policy(transcript).split()

        if rng.random() < error_rate:
            if toks[0] == "<answer>":
                toks[1] = (
                    str((int(toks[1]) + 1 + rng.randrange(5)) % 100)
                    if toks[1].isdigit() else "0"
                )

            elif toks[1] == "lookup":
                toks[2] = rng.choice([i for i in ITEMS if i != toks[2]])

            elif rng.random() < 0.5 and toks[2].isdigit():
                toks[2] = str(int(toks[2]) + 1)

            else:
                toks = toks[:-1]

        return " ".join(toks)

    return policy

# ---------------------------------------------------------------- end of provided code


def run_tool(name, args, store):
    """
    Execute one tool call and return its output as a string.

    Supported tools:
        calc   : Performs addition, subtraction, or multiplication.
        lookup : Retrieves an item's stock quantity from the store.

    Returns:
        str: The calculated result, stock quantity, or "error".

    The tool never allows an exception to propagate to the agent.
    """

    try:
        # Validate the tool name and argument container.
        if not isinstance(name, str) or not isinstance(args, list):
            return "error"

        # --------------------------------------------------
        # Calculator tool
        # --------------------------------------------------
        if name == "calc":

            # The calculator requires exactly three arguments:
            # [first_number, operator, second_number].
            if len(args) != 3:
                return "error"

            a, op, b = args

            # Both operands must be strings containing digits only.
            if not isinstance(a, str) or not a.isdigit():
                return "error"

            if not isinstance(b, str) or not b.isdigit():
                return "error"

            # Only the three specified arithmetic operators are valid.
            if not isinstance(op, str) or op not in ("+", "-", "*"):
                return "error"

            # Convert the operands into integers.
            a = int(a)
            b = int(b)

            # Perform the requested arithmetic operation.
            if op == "+":
                result = a + b

            elif op == "-":
                result = a - b

            else:  # op == "*"
                result = a * b

            # Always return the result as a string.
            return str(result)

        # --------------------------------------------------
        # Lookup tool
        # --------------------------------------------------
        elif name == "lookup":

            # The lookup tool requires exactly one argument.
            if len(args) != 1:
                return "error"

            key = args[0]

            # The lookup key must be a string and must exist in the store.
            if not isinstance(key, str) or key not in store:
                return "error"

            # Retrieve the stock quantity and return it as a string.
            return str(store[key])

        # --------------------------------------------------
        # Invalid or unsupported tool
        # --------------------------------------------------
        else:
            return "error"

    except Exception:
        # Handle unexpected errors gracefully.
        # A tool must never raise an exception.
        return "error"

# Step 2 - parse_action
def parse_action(text):
    """
    Parse the model-generated text and return its first action.

    Supported formats:
        <call> name arg1 arg2 ... </call>
        <answer> value </answer>

    Returns:
        A dictionary representing a tool call, final answer,
        or invalid action.
    """

    # Handle empty or non-string input.
    if not isinstance(text, str):
        return {"type": "invalid"}

    # Split the input text into whitespace-separated tokens.
    tokens = text.split()

    # Empty input or an invalid opening tag.
    if not tokens or tokens[0] not in ("<call>", "<answer>"):
        return {"type": "invalid"}

    # Determine the action type and its corresponding closing tag.
    action_type = tokens[0]

    if action_type == "<call>":
        closing_tag = "</call>"
    else:
        closing_tag = "</answer>"

    # Find the closing tag of the first action.
    try:
        end_index = tokens.index(closing_tag, 1)
    except ValueError:
        # The required closing tag is missing.
        return {"type": "invalid"}

    # Extract tokens between the opening and closing tags.
    content = tokens[1:end_index]

    # Reject any token that begins with '<' inside the action.
    if any(token.startswith("<") for token in content):
        return {"type": "invalid"}

    # --------------------------------------------------
    # Parse tool call
    # --------------------------------------------------
    if action_type == "<call>":

        # A tool call must contain a tool name.
        if len(content) < 1:
            return {"type": "invalid"}

        # The first token is the tool name.
        name = content[0]

        # Remaining tokens are the tool arguments.
        args = content[1:]

        return {
            "type": "call",
            "name": name,
            "args": args
        }

    # --------------------------------------------------
    # Parse final answer
    # --------------------------------------------------
    elif action_type == "<answer>":

        # A final answer must contain exactly one token.
        if len(content) != 1:
            return {"type": "invalid"}

        return {
            "type": "answer",
            "value": content[0]
        }

    # Any unsupported action is invalid.
    return {"type": "invalid"}

# Step 3 - run_episode
def run_episode(policy, task, store, max_calls=4):
    """
    Run one task using a tool-using agent and evaluate its final answer.

    Args:
        policy: A function that receives the transcript and returns
                the model's next generated text.
        task: A dictionary containing the question.
        store: The warehouse dictionary used by the lookup tool.
        max_calls: Maximum number of tool calls allowed.

    Returns:
        A dictionary containing:
            status      : "answered", "invalid", or "budget"
            answer      : The final answer token, or None.
            correct     : Whether the final answer is correct.
            calls       : List of (name, args, result) tuples.
            transcript  : The complete interaction transcript.
    """

    # Initialize the conversation transcript with the question.
    transcript = "<q> " + task["question"] + " </q>"

    # Store all executed tool calls and their results.
    calls = []

    # Initialize the final answer.
    answer = None

    # Default status in case no valid action is completed.
    status = "invalid"

    # Continue interacting with the model until the episode terminates.
    while True:

        # Ask the policy for its next action using the current transcript.
        model_output = policy(transcript)

        # Parse the model's generated text into a structured action.
        action = parse_action(model_output)

        # --------------------------------------------------
        # Case 1: Invalid action
        # --------------------------------------------------
        if action["type"] == "invalid":

            status = "invalid"
            break

        # --------------------------------------------------
        # Case 2: Final answer
        # --------------------------------------------------
        elif action["type"] == "answer":

            # Extract the answer token.
            answer = action["value"]

            # Rebuild the transcript using only the parsed answer.
            transcript += " <answer> " + answer + " </answer>"

            status = "answered"
            break

        # --------------------------------------------------
        # Case 3: Tool call
        # --------------------------------------------------
        elif action["type"] == "call":

            # Check whether the tool-call budget has been exhausted.
            if len(calls) >= max_calls:

                status = "budget"
                break

            # Extract the tool name and its arguments.
            name = action["name"]
            args = action["args"]

            # Execute the requested tool.
            result = run_tool(name, args, store)

            # Record the tool call, its arguments, and its result.
            calls.append((name, args, result))

            # Rebuild the transcript using the parsed tool call
            # and the tool-generated result.
            call_text = " ".join([name] + args)

            transcript += (
                " <call> " + call_text + " </call>"
                + " <result> " + result + " </result>"
            )

            # Continue so the policy can observe the tool result.
            continue

    # --------------------------------------------------
    # Grade the final answer
    # --------------------------------------------------

    # Calculate the correct answer directly using the provided checker.
    expected_answer = str(solve(task["question"], store))

    # An episode is correct only if the model answered and
    # its answer exactly matches the expected answer.
    correct = (status == "answered" and answer == expected_answer)

    # Return all episode information.
    return {
        "status": status,
        "answer": answer,
        "correct": correct,
        "calls": calls,
        "transcript": transcript
    }

# Step 4 - collect_trajectories
def collect_trajectories(tasks, stores, teacher):
    """
    Run the teacher on every task using its corresponding store.

    Retain only episodes that produce correct answers.

    Args:
        tasks: List of task dictionaries.
        stores: List of warehouse dictionaries, one per task.
        teacher: A policy function that generates the next action.

    Returns:
        A tuple containing:
            kept: List of accepted training records.
            dropped: Number of rejected episodes.
    """

    # Store the successfully completed and verified trajectories.
    kept = []

    # Count episodes that fail the correctness checker.
    dropped = 0

    # Process every task using its corresponding warehouse.
    for task, store in zip(tasks, stores):

        # Run the teacher policy on the current task.
        # The default tool-call budget is used.
        episode = run_episode(teacher, task, store)

        # Retain the trajectory only if the answer checker accepts it.
        if episode["correct"]:

            kept.append({
                "id": task["id"],
                "question": task["question"],
                "text": episode["transcript"]
            })

        else:
            # Reject incorrect, invalid, or unfinished episodes.
            dropped += 1

    # Return accepted trajectories and the number of rejected episodes.
    return kept, dropped

# Step 5 - encode_trajectory
def encode_trajectory(text, block_size=40):
    """
    Convert a trajectory into token IDs and a loss mask.

    The loss mask:
        1 = Token generated by the model (<call> or <answer> spans).
        0 = Token provided to the model (question or tool results).

    Both lists are padded to block_size.

    Args:
        text: The complete trajectory as a string.
        block_size: Maximum sequence length.

    Returns:
        ids: List of token IDs.
        mask: List of binary loss-mask values.
    """

    # Split the trajectory into whitespace-separated tokens.
    tokens = text.split()

    # Ensure the trajectory does not exceed the allowed sequence length.
    if len(tokens) > block_size:
        raise ValueError("Trajectory exceeds block_size.")

    # Convert tokens into their corresponding vocabulary IDs.
    # Unknown tokens are mapped to the <unk> token ID.
    ids = [
        STOI.get(token, STOI["<unk>"])
        for token in tokens
    ]

    # Initialize the loss mask with zeros.
    mask = [0] * len(tokens)

    # Track the closing tag of the current model-generated span.
    active_end = None

    # Identify tokens generated by the model.
    for i, token in enumerate(tokens):

        # Detect the beginning of a model-generated tool call.
        if active_end is None:

            if token == "<call>":
                active_end = "</call>"

            # Detect the beginning of a model-generated final answer.
            elif token == "<answer>":
                active_end = "</answer>"

        # Mark the complete call or answer span as model-generated.
        # Opening and closing tags are included in the loss.
        if active_end is not None:
            mask[i] = 1

            # End the model-generated span at its closing tag.
            if token == active_end:
                active_end = None

    # Pad token IDs with the <pad> token ID.
    pad_id = STOI["<pad>"]
    padding_length = block_size - len(ids)

    ids.extend([pad_id] * padding_length)

    # Pad the loss mask with zeros.
    mask.extend([0] * padding_length)

    return ids, mask

# Step 6 - ToolLM
import torch
import torch.nn as nn


class ToolLM(nn.Module):
    """
    A small causal Transformer language model.

    Input:
        Token IDs of shape (B, T).

    Output:
        Next-token logits of shape (B, T, vocab_size).
    """

    def __init__(self, vocab_size, block_size=40, d_model=48, n_head=4, n_layer=1):
        super().__init__()

        # Maximum supported sequence length.
        self.block_size = block_size

        # Token embeddings: convert vocabulary IDs into dense vectors.
        self.tok_emb = nn.Embedding(vocab_size, d_model)

        # Positional embeddings: encode the position of each token.
        self.pos_emb = nn.Embedding(block_size, d_model)

        # Stack of causal Transformer encoder layers.
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=4 * d_model,
                dropout=0.0,
                batch_first=True,
                norm_first=True
            )
            for _ in range(n_layer)
        ])

        # Final layer normalization.
        self.ln_f = nn.LayerNorm(d_model)

        # Linear projection from hidden dimension to vocabulary logits.
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        """
        Forward pass through the causal Transformer.

        Args:
            idx: Input token IDs with shape (B, T).

        Returns:
            Logits with shape (B, T, vocab_size).
        """

        # Extract batch size and sequence length.
        B, T = idx.shape

        # Ensure the sequence does not exceed the configured block size.
        if T > self.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds block_size {self.block_size}."
            )

        # Create position indices: 0, 1, 2, ..., T-1.
        positions = torch.arange(T, device=idx.device)

        # Obtain token embeddings and positional embeddings.
        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb(positions)

        # Combine token and positional information.
        x = tok_emb + pos_emb.unsqueeze(0)

        # Create a causal attention mask.
        # True above the diagonal prevents attending to future positions.
        causal = torch.triu(
            torch.ones(
                (T, T),
                dtype=torch.bool,
                device=idx.device
            ),
            diagonal=1
        )

        # Pass the representations through every Transformer block.
        for block in self.blocks:
            x = block(x, src_mask=causal)

        # Apply final layer normalization.
        x = self.ln_f(x)

        # Project hidden representations into vocabulary logits.
        logits = self.head(x)

        return logits

# Step 7 - sft_loss
import torch.nn.functional as F


def sft_loss(logits, ids, mask):
    """
    Calculate the mean next-token cross-entropy loss
    over model-generated tokens only.

    Args:
        logits: Model output of shape (B, T, V).
        ids: Token IDs of shape (B, T).
        mask: Binary loss mask of shape (B, T).

    Returns:
        A scalar tensor containing the masked cross-entropy loss.
    """

    # Shift logits and token IDs by one position.
    # Logits at position t predict the token at position t + 1.
    pred_logits = logits[:, :-1, :]
    targets = ids[:, 1:]

    # Shift the loss mask to align with the target tokens.
    target_mask = mask[:, 1:]

    # Get the vocabulary size.
    vocab_size = logits.size(-1)

    # Calculate cross-entropy for every next-token prediction.
    # reduction='none' preserves the individual token losses.
    losses = F.cross_entropy(
        pred_logits.reshape(-1, vocab_size),
        targets.reshape(-1),
        reduction="none"
    )

    # Convert the mask to the same device and dtype as the losses.
    target_mask = target_mask.reshape(-1).to(
        device=losses.device,
        dtype=losses.dtype
    )

    # Keep losses only for tokens generated by the model.
    masked_losses = losses * target_mask

    # Count the model-generated target tokens across the entire batch.
    num_targets = target_mask.sum()

    # Calculate the mean loss over selected target tokens.
    # clamp_min prevents division by zero if no targets are selected.
    loss = masked_losses.sum() / num_targets.clamp_min(1.0)

    return loss

# Step 8 - train_sft
def train_sft(model, ids, mask, steps, batch_size=32, lr=3e-3, generator=None):
    """
    Fine-tune the model on encoded trajectories using masked SFT loss.

    Args:
        model: The ToolLM model to train.
        ids: Token ID tensor of shape (N, T).
        mask: Loss mask tensor of shape (N, T).
        steps: Number of training iterations.
        batch_size: Number of trajectories sampled per iteration.
        lr: Learning rate for AdamW.
        generator: Optional PyTorch random number generator.

    Returns:
        A list containing the training loss from each step as floats.
    """

    # Create the AdamW optimizer once for the entire training process.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr
    )

    # Get the total number of available trajectories.
    N = ids.size(0)

    # Store the loss from every training iteration.
    losses = []

    # Perform the specified number of training steps.
    for step in range(steps):

        # Randomly sample trajectory indices with replacement.
        rows = torch.randint(
            0,
            N,
            (batch_size,),
            generator=generator
        )

        # Select the corresponding token IDs and loss masks.
        batch_ids = ids[rows]
        batch_mask = mask[rows]

        # Forward pass: predict the next token at every position.
        logits = model(batch_ids)

        # Calculate the masked supervised fine-tuning loss.
        loss = sft_loss(
            logits,
            batch_ids,
            batch_mask
        )

        # Clear gradients from the previous training iteration.
        optimizer.zero_grad(set_to_none=True)

        # Backpropagate the loss to calculate gradients.
        loss.backward()

        # Update the model parameters.
        optimizer.step()

        # Record the current loss as a Python float.
        losses.append(loss.item())

    return losses

# Step 9 - model_policy (not yet solved)
# TODO: implement

# Step 10 - run_trials (not yet solved)
# TODO: implement

# Step 11 - pass_k (not yet solved)
# TODO: implement

# Step 12 - score_with_se (not yet solved)
# TODO: implement

# Step 13 - compare_checkpoints (not yet solved)
# TODO: implement

# Step 14 - contamination_report (not yet solved)
# TODO: implement

