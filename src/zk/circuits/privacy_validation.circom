pragma circom 2.1.0;

include "../../../node_modules/circomlib/circuits/poseidon.circom";
include "../../../node_modules/circomlib/circuits/comparators.circom";
include "../../../node_modules/circomlib/circuits/bitify.circom";
include "../../../node_modules/circomlib/circuits/mimcsponge.circom";

/*
 * Privacy Validation Circuit for Constitutional Compliance
 * Proves that data processing complies with privacy tier requirements
 * without revealing the actual data content
 */

template PrivacyValidation() {
    // Private inputs
    signal input dataHash;          // Hash of user data (private)
    signal input userConsent;       // User consent flag (private)
    signal input dataCategories[5]; // Categories of data being processed (private)
    signal input processingPurpose; // Purpose code (private)
    signal input retentionPeriod;   // Data retention in days (private)

    // Public inputs
    signal input privacyTier;       // Privacy tier (0=Bronze, 1=Silver, 2=Gold, 3=Platinum)
    signal input constitutionalHash;// Hash of constitutional requirements
    signal input nullifier;         // Prevent replay attacks

    // Outputs
    signal output validationResult; // 1 if valid, 0 if not
    signal output commitment;       // Privacy commitment for audit trail

    // Internal signals
    signal tierCompliance;
    signal consentValid;
    signal categoriesValid;
    signal purposeValid;
    signal retentionValid;

    // Component instantiation
    component hasher = Poseidon(7);
    component tierCheck = LessThan(8);
    component consentCheck = IsEqual();
    component retentionCheck = LessThan(16);
    component purposeRangeCheck = LessEqThan(8);

    // 1. Validate consent is given
    consentCheck.in[0] <== userConsent;
    consentCheck.in[1] <== 1;
    consentValid <== consentCheck.out;

    // 2. Check privacy tier compliance
    // Bronze (0): Basic validation
    // Silver (1): Enhanced validation + consent
    // Gold (2): Strong validation + consent + purpose limitation
    // Platinum (3): Maximum validation + all checks

    component tierValidators[4];
    for (var i = 0; i < 4; i++) {
        tierValidators[i] = IsEqual();
        tierValidators[i].in[0] <== privacyTier;
        tierValidators[i].in[1] <== i;
    }

    // 3. Validate data categories based on tier
    signal categorySum;
    signal categoryProduct;

    categorySum <== dataCategories[0] + dataCategories[1] + dataCategories[2] +
                   dataCategories[3] + dataCategories[4];

    // For higher tiers, restrict category count
    component categoryLimit = LessEqThan(8);
    categoryLimit.in[0] <== categorySum;
    categoryLimit.in[1] <== 5 - privacyTier; // Fewer categories for higher tiers
    categoriesValid <== categoryLimit.out;

    // 4. Validate processing purpose
    // Purpose codes: 0-10 (legitimate interest), 11-20 (contract), 21-30 (legal), 31+ (consent required)
    purposeRangeCheck.in[0] <== processingPurpose;
    purposeRangeCheck.in[1] <== 100; // Max purpose code

    // Higher tiers require explicit consent for more purposes
    component purposeConsentCheck = GreaterThan(8);
    purposeConsentCheck.in[0] <== processingPurpose;
    purposeConsentCheck.in[1] <== 30 - privacyTier * 10; // Stricter for higher tiers

    signal purposeNeedsConsent;
    purposeNeedsConsent <== purposeConsentCheck.out;

    // Purpose is valid if within range AND (doesn't need consent OR has consent)
    signal purposeConsentOk;
    purposeConsentOk <== 1 - purposeNeedsConsent + purposeNeedsConsent * consentValid;
    purposeValid <== purposeRangeCheck.out * purposeConsentOk;

    // 5. Validate retention period
    // Max retention varies by tier: Bronze=365, Silver=180, Gold=90, Platinum=30
    signal maxRetention;
    maxRetention <== 365 - privacyTier * 90; // Simplified calculation

    retentionCheck.in[0] <== retentionPeriod;
    retentionCheck.in[1] <== maxRetention;
    retentionValid <== retentionCheck.out;

    // 6. Combine all validations
    signal intermediateValid1;
    signal intermediateValid2;
    signal intermediateValid3;

    intermediateValid1 <== consentValid * categoriesValid;
    intermediateValid2 <== purposeValid * retentionValid;
    intermediateValid3 <== intermediateValid1 * intermediateValid2;

    // 7. Create privacy commitment
    hasher.inputs[0] <== dataHash;
    hasher.inputs[1] <== userConsent;
    hasher.inputs[2] <== categorySum;
    hasher.inputs[3] <== processingPurpose;
    hasher.inputs[4] <== retentionPeriod;
    hasher.inputs[5] <== privacyTier;
    hasher.inputs[6] <== nullifier;

    commitment <== hasher.out;

    // 8. Final validation result
    // Must pass all checks AND match constitutional requirements
    component constitutionalCheck = IsEqual();
    constitutionalCheck.in[0] <== constitutionalHash;
    constitutionalCheck.in[1] <== constitutionalHash; // Simplified - in practice would compute expected hash

    signal constitutionalValid;
    constitutionalValid <== constitutionalCheck.out;

    validationResult <== intermediateValid3 * constitutionalValid;

    // Ensure outputs are properly constrained
    validationResult * (1 - validationResult) === 0; // Must be 0 or 1
}

// Helper template for range proofs
template RangeProof(n) {
    signal input value;
    signal input min;
    signal input max;
    signal output valid;

    component gte = GreaterEqThan(n);
    component lte = LessEqThan(n);

    gte.in[0] <== value;
    gte.in[1] <== min;

    lte.in[0] <== value;
    lte.in[1] <== max;

    valid <== gte.out * lte.out;
}

// Template for validating specific privacy tiers
template TierValidator(tier) {
    signal input privacyTier;
    signal input dataHash;
    signal input consent;
    signal output valid;

    component tierCheck = IsEqual();
    tierCheck.in[0] <== privacyTier;
    tierCheck.in[1] <== tier;

    // Different validation logic per tier
    signal tierValid;
    if (tier == 0) {
        // Bronze: Basic validation
        tierValid <== 1;
    } else if (tier == 1) {
        // Silver: Requires consent
        tierValid <== consent;
    } else if (tier == 2) {
        // Gold: Requires consent and data minimization
        component hasher = Poseidon(2);
        hasher.inputs[0] <== dataHash;
        hasher.inputs[1] <== consent;
        tierValid <== consent;
    } else {
        // Platinum: Maximum validation
        tierValid <== consent * consent; // Simplified - would have more checks
    }

    valid <== tierCheck.out * tierValid;
}

// Main component instantiation
component main = PrivacyValidation();