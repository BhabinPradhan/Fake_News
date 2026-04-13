# MOSAIC User Guide

> A guide to using the MOSAIC fake news detection interface. Learn what to enter, how to interpret results, and what the system is telling you.

MOSAIC analyzes news content using 28 specialized AI models working together. It examines both the text and image from a post to determine if it's Real, Fake, or Uncertain. This guide will help you understand how to use the system and what the results mean.

---

## What You Need

To use MOSAIC, you must provide **both** of the following:

- **Text:** The caption, headline, or article content from the news post
- **Image:** The photo or graphic that accompanied the post

**Why both are required:** MOSAIC uses multimodal AI models specifically trained to detect fake news by analyzing how text and images work together. Without both components, the analysis cannot be performed.

---

## Step 1: Enter Your Content

### Text Input

Paste the full text of the news article or social media post into the text field.

**What to enter:**

- Complete news articles
- Social media captions (Facebook, Twitter, Instagram)
- Headlines with body text
- Any content claiming to be factual news

**Tips for best results:**

- Include the full text when possible (not just headlines)
- Copy exactly as written—do not paraphrase
- More text generally improves accuracy (aim for multiple sentences)

**Example:**

    BREAKING: Scientists discover cure for all cancers. 
    Pharmaceutical companies don't want you to know! 
    Major hospital confirms 100% success rate in trials. 
    Share before this gets deleted!

### Image Upload

Click the upload button and select the image that came with the news post.

**What images to upload:**

-  Photos accompanying the news story
-  Screenshots of social media posts
-  Infographics or charts from the article
-  Any visual content presented with the text

**What NOT to upload:**

-  Random unrelated images
- Stock photos you added yourself

**Why images matter:**

- Fake news often uses misleading, doctored, or mismatched images
- The relationship between text and image reveals manipulation tactics
- Image metadata can expose timeline inconsistencies

---

## Step 2: Click "Analyze"

Once you have entered both text and uploaded an image, click the **"Analyze"** button.

**What happens:**

- Your content is sent to the ensemble of 28 AI models
- Each model analyzes different aspects (text patterns, image authenticity, multimodal alignment)
- Results typically appear within 5–15 seconds
- A progress indicator shows the system is working

**Note:** The "Analyze" button will be disabled until both text and image are provided.

---

## Step 3: Understanding Your Results

### The Three Possible Verdicts

MOSAIC returns one of three verdicts based on how the 28 expert models voted:

####  Real

**What it means:**

- The content is likely legitimate news
- The majority of models (or all models) classified it as real
- High confidence in the assessment

**What to do:**

- The content appears trustworthy
- Still verify important claims using multiple sources
- Check for updates if the story is breaking news

---

#### Fake

**What it means:**

- The content is likely misinformation or fabricated news
- The majority of models (or all models) classified it as fake
- High confidence in the assessment

**What to do:**

- **Do not share** this content
- Be skeptical of the claims presented
- Report the post if on social media

---

#### Uncertain

**What it means:**

- Models disagree about whether the content is real or fake
- Mixed signals detected in the text and/or image
- Insufficient confidence to make a definitive call

**What to do:**

- Treat with significant caution
- Perform additional fact-checking before sharing or believing
- Use fact-checking websites (Snopes, FactCheck.org, PolitiFact)
- Search for the story on established news outlets

---

### What "Uncertain" Really Means

When MOSAIC returns **Uncertain**, it indicates one or more of the following:

| Reason | Explanation |
| :--- | :--- |
| **Model Disagreement** | Some models say Real, others say Fake |
| **Mixed Signals** | Content has characteristics of both real and fake news |
| **Insufficient Information** | Not enough context to make a confident decision |
| **Ambiguous Writing** | The tone and style are neutral or unclear |
| **Text-Image Mismatch** | The image and text may be telling different stories |

**This is not a system failure.** Uncertain verdicts reflect honest limitations. Real-world content isn't always clearly real or clearly fake.

**Best practice when you see Uncertain:**

1. Look up the story on fact-checking sites (Snopes, FactCheck.org, PolitiFact)
2. Search for coverage from established, credible news outlets
3. Check whether claims cite verifiable sources
4. Be extra cautious—**do not share without verification**
5. Do not assume real just because it wasn't flagged as fake

---

## Viewing Options: Simple vs. Technical

MOSAIC offers two viewing modes. Toggle between them using the button at the top of the results.

### Simple View (Recommended)

**What you see:**

- Clear verdict (Real, Fake, or Uncertain)
- Overall confidence percentage
- Vote strength and agreement metrics
- Plain-language summary
- Key reasons for the decision

**Best for:**

- Quick fact-checking
- Sharing results with non-technical audiences
- When you just need the answer
- General users

---

### Technical View (For Advanced Users)

**What you see:**

- Individual predictions from all 28 models
- Breakdown by model family (MoPeD, COOLANT, MCAN, etc.)
- Per-model confidence scores
- Statistical agreement and vote strength
- Technical reasoning from each expert

**Best for:**

- Researchers and students
- Understanding the ensemble decision-making process
- Comparing how different model types perform
- Detailed justification for the verdict


---

## Understanding the XAI Card

The **Explainable AI (XAI) Card** explains **why** MOSAIC made its decision. This transparency helps you understand and critically evaluate the results.

### What the XAI Card Shows

#### 1. Key Indicators Found

The system highlights patterns detected in the content:

**Text Analysis:**

| Indicator | Type | Meaning |
| :--- | :--- | :--- |
|  Sensational Language | Fake Signal | ALL CAPS, "SHOCKING!", urgency tactics |
|  Unverified Claims | Fake Signal | Statements without credible sources |
|  Clickbait Patterns | Fake Signal | "You won't believe...", "What happens next..." |
|  Conspiracy Rhetoric | Fake Signal | "They don't want you to know", "Hidden truth" |
|  Neutral Tone | Real Signal | Professional, factual writing style |
|  Credible Sources | Real Signal | Named organizations, studies, experts |
|  Verifiable Facts | Real Signal | Claims that can be independently checked |

**Image Analysis:**

| Indicator | Type | Meaning |
| :--- | :--- | :--- |
|  Possible Manipulation | Fake Signal | Signs of photo editing or doctoring |
|  Mismatched Context | Fake Signal | Image doesn't match the story content |
|  Low Quality | Fake Signal | Heavily compressed (often re-shared many times) |
|  Metadata Inconsistency | Fake Signal | Image date doesn't match claimed event date |
|  Original Source Found | Real Signal | Image traced to legitimate source |
|  Contextually Appropriate | Real Signal | Photo matches story content |

---

#### 2. Confidence Scores

MOSAIC reports confidence as a percentage. Here's how to interpret it:

| Confidence Range | Interpretation | Recommendation |
| :--- | :--- | :--- |
| **80–100%** | High confidence; multiple models strongly agree | Trust this assessment |
| **60–79%** | Medium confidence; most models agree | Likely accurate; verify if important |
| **Below 60%** | Low confidence; models disagree or uncertain | Result shown as "Uncertain"; fact-check |

---

#### 3. Supporting Experts

The XAI card lists which specific models contributed most strongly to the verdict.

#### 4. Most Influential Factors

This section highlights **what mattered most** in the decision.


### How to Use the XAI Card

**DO:**

- Read the key indicators to understand the reasoning
- Look for multiple red flags (more flags = higher likelihood of fake)
- Consider whether the explanation makes logical sense
- Use it to learn what patterns indicate fake vs. real news

**DON'T:**

- Ignore the XAI card and blindly trust the verdict
- Dismiss content as fake based on only one indicator
- Share results without understanding the reasoning

---

## Key Metrics Explained

MOSAIC reports several metrics alongside the verdict. Here's what they mean:

| Metric | Range | Meaning |
| :--- | :--- | :--- |
| **Confidence** | 0–100% | How certain the system is in its verdict |
| **Agreement** | 0.0–1.0 | What fraction of models agree with the majority vote |
| **Vote Strength** | 0.0–1.0 | How strongly models lean toward their verdict (not just binary) |

**Example interpretation:**

    Confidence: 87%
    Agreement: 0.82
    Vote Strength: 0.64

- **87% confidence:** High certainty in the verdict
- **0.82 agreement:** 82% of models voted with the majority
- **0.64 vote strength:** Models moderately favor their prediction (not extreme)

---

## Understanding System Limitations

### What MOSAIC Can Do:

-  Detect common fake news patterns and manipulation tactics
-  Identify sensational or emotionally manipulative language
-  Spot mismatched or misleading images
-  Recognize credible source citations
-  Analyze multiple signals simultaneously across 28 models

---

### What MOSAIC Cannot Do:

-  Verify specific factual claims (e.g., "Event X happened on Date Y")
-  Access real-time information (cannot check if something happened today)
-  Determine satirical intent (The Onion articles may be flagged as fake)
-  Replace human judgment and critical thinking
-  Guarantee 100% accuracy (AI systems make mistakes)

---

### When to Be Extra Careful:

| Content Type | Why Extra Care Is Needed |
| :--- | :--- |
| **Political content** | High stakes, often controversial and polarizing |
| **Health/medical claims** | Can cause real harm if misinformation spreads |
| **Breaking news** | Not enough time for proper verification |
| **Unknown sources** | No track record of credibility |
| **Too good/bad to be true** | Often designed to trigger emotional sharing |



## Quick Reference Table

| Verdict | Confidence | What It Means | Recommended Action |
| :--- | :--- | :--- | :--- |
|  **REAL** | 90%+ | Very likely legitimate | Still cross-check important claims |
|  **REAL** | 70–89% | Probably legitimate | Verify before sharing widely |
|  **UNCERTAIN** | 50–69% | Mixed signals, unclear | Perform thorough fact-checking |
|  **UNCERTAIN** | <50% | Not enough information or models split | Find more complete source; verify independently |
|  **FAKE** | 70–89% | Probably misinformation | Don't share; be skeptical |
|  **FAKE** | 90%+ | Very likely misinformation | Definitely don't share; report if on social media |

---

## Frequently Asked Questions

### Can I trust MOSAIC 100%?

**No.** No AI system is perfect. MOSAIC is a powerful tool, but always combine it with:

- Your own critical thinking
- Fact-checking websites (Snopes, FactCheck.org, PolitiFact)
- Checking multiple credible news sources
- Looking for original sources of claims

---

### Why does it sometimes say "Uncertain"?

**Because honest uncertainty is better than a wrong answer.** When models disagree or don't have enough information, MOSAIC tells you to be extra careful rather than guessing.

---

### What if I disagree with the result?

That's completely fine. MOSAIC is a tool to help you, not replace your judgment. If you have good reasons to disagree:

1. Check the XAI explanation—did it miss something important?
2. Perform your own fact-checking
3. Consider providing feedback (if available in your deployment)

---

### Can it detect satire (like The Onion)?

**Not reliably.** Satire is intentionally written to mimic real news, so it may be flagged as fake. Use context clues and check the source website to identify satire.

---

### How does it analyze images?

MOSAIC's image analysis looks for:

- Signs of photo manipulation or editing
- Whether the image matches the text content
- If the image has been used in other contexts (reverse image search)
- Image quality and metadata (date, location if available)

---

### What if the news post I saw didn't have an image?

MOSAIC requires both text and an image because it's a multimodal system. If the post has no image:

1. Try to find the original source—often the full article includes images
2. Look for the story on other news sites that include images
3. If no image exists anywhere, MOSAIC won't be able to analyze this post
4. Use traditional fact-checking methods instead (verifying sources, searching on Snopes, etc.)

---

### What's the difference between the AI models shown in Technical View?

Each model uses different techniques and was trained on different datasets:

- **MoPeD, COOLANT, MCAN:** Trained on specific fake news datasets (Snopes, Weibo, XFacta)
- **ATT-RNN, SpotFake:** Use different neural network architectures
- **MVAE:** Focuses on multimodal alignment between text and images

The **ensemble** combines all their predictions to make a more robust final decision.

---

### Does it work in languages other than English?

Check with your system administrator. Some models support multiple languages, particularly:

- **English (EN):** Fully supported
- **Chinese (ZH):** Supported via Weibo-trained models
- Other languages may have limited or no support

---

## Tips for Evaluating News (Beyond MOSAIC)

Even with MOSAIC, develop these critical thinking habits:

###  Check the Source

- Is this from a known, credible news outlet?
- Can you find the original source of the claim?
- Does the website have an "About" page?
- Is the source known for fact-based reporting?

---

###  Look for Evidence

- Are sources cited for major claims?
- Can you verify the sources exist and said what's claimed?
- Are there links to original documents or studies?
- Do quotes match what the person actually said?

---

###  Check the Date

- Is this current news or old content being recirculated?
- Does the image date match the story date?
- Has the situation changed since the article was written?

---

###  Read Beyond Headlines

- Does the article actually support what the headline claims?
- Is the content more nuanced than the headline suggests?
- Are you being misled by clickbait?

---

###  Check Your Emotions

- Does this make you very angry, scared, or excited?
- Could it be designed to manipulate your emotions?
- Would you believe this if you weren't emotional about it?

---

###  Look for Corroboration

- Are other credible outlets reporting this story?
- If it's true and important, there should be multiple sources
- Be suspicious of claims no one else is reporting

---

## Remember

**MOSAIC is a tool to help you think critically, not to think for you.**

The goal is to:

- Make you more aware of fake news tactics and manipulation
- Help you evaluate content more carefully and systematically
- Save you time by flagging suspicious content
- Teach you what patterns to look for in future content

**You are still responsible for:**

- Verifying important information before acting on it
- Using good judgment and critical thinking
- Not sharing unverified or questionable content
- Thinking carefully about what you read and believe

---

*Fighting misinformation starts with each of us taking responsibility for what we share. MOSAIC is here to help you be part of the solution.*
