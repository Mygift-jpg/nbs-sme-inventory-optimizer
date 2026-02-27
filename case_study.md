# 🇳🇬 NBS SME Inventory Optimizer: A 4-Week Journey

## The Problem
Lagos rice sellers struggle to know WHEN to buy. Prices fluctuate with FX, seasons, and market shocks. Most rely on guesswork.

## The Goal
Build a simple tool that tells SMEs: "Buy now" or "Wait" — based on real data.

## The Data
- 2023-2024: NBS Nigeria food price data
- 2025-2026: Extended using FX rates + market trends
- 31 months of rice prices (₦/50kg bag)

## The Model
Linear Regression (chosen for transparency, not black box)

### Key Drivers (₦ Impact)
- Festive Season: **+₦2,714**
- Shock Month: **+₦984**
- Price Momentum: **+₦819 per 1%**
- Inflation: **+₦808 per 1%**
- Harvest Season: **-₦708**

## The Struggles
❌ Random Forest overfitted (R² = -30)
❌ NaN errors in training data
❌ Feature mismatches (10 vs 7)
❌ Model stuck at ₦102,022 for weeks
❌ Almost quit multiple times

## The Breakthroughs
✅ Switched to Linear Regression
✅ Removed time trend (months_since_start)
✅ Manually fixed sign errors (Festive + Shock)
✅ Intercept tuning: ₦56,876 March price ✅

## The Final Product
📍 Live at: https://lnkd.in/d4dfw6Er

Features:
- 📊 FX Scenario Simulator
- 📈 Sensitivity Chart
- ⚡ Quick Scenarios
- 🔑 Key Insights
- 📋 Full transparency

## The Impact
SMEs can now:
- Test "what if" scenarios
- See their risk instantly
- Make data-driven decisions

## What's Next? (March)
- [ ] Add more products (beans, garri, oil)
- [ ] User feedback collection
- [ ] Mobile-friendly improvements
- [ ] Deploy to more SMEs

## Built By
Blessing Okagbare  
#PROOF2026 • Building in Public • 🇳🇬