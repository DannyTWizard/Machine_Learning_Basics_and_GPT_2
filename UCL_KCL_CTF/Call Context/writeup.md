# Call Context - CTF Challenge Writeup

## Challenge Description
Analyze a transcript of a phone conversation and identify the city being discussed based on contextual clues.

## Original Message (Transcript)

```
[February 11, 2023]

[???]
I arrived earlier today. The journey took longer than expected, but everything was organised surprisingly well.

[???]
Is it busy already?

[???]
Very. I didn't realise quite how many people would be here at once.

[???]
That doesn't surprise me. This time of year is always like that.

[???]
There's a large religious site near where I'm staying. I walked past it this afternoon and it was packed.

[???]
Yes, that's the main reason people come there.

[???]
I felt a bit out of place at first. So many groups, guides, and volunteers everywhere.

[???]
How are you managing with the language?

[???]
French mostly. English works sometimes, but not with everyone.

[???]
Are you staying long?

[???]
Just a few days. Most people seem to be passing through rather than living here.

[???]
Will you visit your uncle while you're nearby?

[???]
He lives about an hour away, closer to the mountains.

[???]
You're not going hiking?

[???]
No, not really. I can already see them from here anyway.

[???]
At least transport is easy.

[???]
Yes, trains and buses seem set up specifically for visitors. I should head back now — I'll message you later.
```

## Analysis

### Key Contextual Clues Identified

| Clue | Interpretation |
|------|----------------|
| **Date: February 11** | February 11 is the Feast of Our Lady of Lourdes, commemorating the first apparition of the Virgin Mary to Bernadette Soubirous in 1858 |
| **"Large religious site... the main reason people come there"** | Major pilgrimage destination - Lourdes is one of the world's most visited Catholic pilgrimage sites |
| **"This time of year is always like that" (busy)** | February 11 attracts many pilgrims for the feast day |
| **"French mostly. English works sometimes"** | French-speaking location with international visitors |
| **"Groups, guides, and volunteers everywhere"** | Organized pilgrimages are common at Lourdes, with volunteer organizations assisting sick and disabled pilgrims |
| **"Most people seem to be passing through rather than living here"** | Lourdes is a small town (population ~14,000) that receives millions of pilgrims annually |
| **"Uncle lives about an hour away, closer to the mountains"** | Located at the foothills of the Pyrenees mountains |
| **"I can already see them from here"** | Mountains visible from the town - Lourdes sits at the base of the Pyrenees |
| **"Trains and buses seem set up specifically for visitors"** | Lourdes has dedicated transport infrastructure for pilgrims, including a TGV station and bus services |

### Conclusion

All contextual clues point definitively to **Lourdes, France**:

- World-famous Catholic pilgrimage site (Sanctuary of Our Lady of Lourdes)
- Located in southwestern France at the foot of the Pyrenees mountains
- The date February 11 is particularly significant as it marks the anniversary of the first apparition
- Known for organized pilgrimage groups with volunteers
- Small town primarily sustained by religious tourism
- Excellent transport links established for pilgrims

## Summary

**City Identified:** Lourdes, France

**Flag:** `flag{lourdes}`
