"""Training"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from src.model import RealEstateDataGenerator, RealEstateModel

print("🏠 Real Estate Analysis Training")
gen = RealEstateDataGenerator()
df = gen.generate_property_data(10000)
df.to_csv('data/properties.csv', index=False)
print(f"✓ Generated {len(df)} properties")
print(f"✓ Avg Price: ${df['price'].mean():,.0f}, Range: ${df['price'].min():,.0f}-${df['price'].max():,.0f}")

model = RealEstateModel()
metrics = model.train(df)
model.save('data/model.pkl')

print(f"\n✓ Model R²: {metrics['r2']:.4f}")
print(f"✓ RMSE: ${metrics['rmse']:,.0f}")
print(f"✓ MAE: ${metrics['mae']:,.0f}")
print("✅ Training complete!")
