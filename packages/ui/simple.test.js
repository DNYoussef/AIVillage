// Simple test to verify Jest is working
describe('Basic Jest Test', () => {
  test('should pass basic test', () => {
    expect(1 + 1).toBe(2);
  });

  test('should verify environment', () => {
    expect(typeof window).toBe('object');
    expect(typeof document).toBe('object');
  });

  test('should handle strings', () => {
    expect('AIVillage').toContain('Village');
    expect('React Test').toHaveLength(10);
  });
});
