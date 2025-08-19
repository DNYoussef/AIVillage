/**
 * Linter C Example
 *
 * Demonstrates code linting and SBOM generation from C
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/betanet.h"

int main() {
    printf("=== Betanet Linter Example ===\n\n");

    // Initialize library
    if (betanet_init() != 0) {
        fprintf(stderr, "Failed to initialize Betanet library\n");
        return 1;
    }

    printf("Library version: %s\n", betanet_version());
    printf("Linter support: %s\n", betanet_feature_supported("linter") == BETANET_SUCCESS ? "Yes" : "No");
    printf("\n");

    // Configure linter
    betanet_LinterConfigFFI config = {
        .target_dir = "../../betanet-htx",  // HTX crate directory
        .min_severity = 1,  // Warning level
        .all_checks = 1     // Enable all checks
    };

    printf("Creating linter with configuration:\n");
    printf("  Target directory: %s\n", config.target_dir);
    printf("  Minimum severity: %s\n", config.min_severity == 0 ? "Info" :
                                       config.min_severity == 1 ? "Warning" :
                                       config.min_severity == 2 ? "Error" : "Critical");
    printf("  All checks: %s\n", config.all_checks ? "enabled" : "disabled");
    printf("\n");

    // Create linter
    LinterHandle* linter = linter_create(&config);
    if (!linter) {
        fprintf(stderr, "Failed to create linter\n");
        betanet_cleanup();
        return 1;
    }

    printf("Linter created successfully!\n");

    // Run linter checks
    printf("\nRunning linter checks...\n");
    betanet_LinterResultsFFI results;
    betanet_Result result = linter_run(linter, &results);

    if (result != betanet_Success) {
        fprintf(stderr, "Failed to run linter: %s\n", betanet_error_message(result));
        linter_free(linter);
        betanet_cleanup();
        return 1;
    }

    printf("Linting complete!\n\n");
    printf("=== Linter Results ===\n");
    printf("Files checked: %u\n", results.files_checked);
    printf("Rules executed: %u\n", results.rules_executed);
    printf("Issues found:\n");
    printf("  Critical: %u\n", results.critical_issues);
    printf("  Error: %u\n", results.error_issues);
    printf("  Warning: %u\n", results.warning_issues);
    printf("  Info: %u\n", results.info_issues);
    printf("Total issues: %u\n", results.critical_issues + results.error_issues +
                                 results.warning_issues + results.info_issues);
    printf("\n");

    // Check specific rule
    printf("Checking specific rule: 'unsafe-code'...\n");
    betanet_LinterResultsFFI rule_results;
    result = linter_check_rule(linter, "unsafe-code", &rule_results);

    if (result == betanet_Success) {
        printf("Unsafe code check results:\n");
        printf("  Files checked: %u\n", rule_results.files_checked);
        printf("  Rules executed: %u\n", rule_results.rules_executed);
        printf("  Issues: %u\n", rule_results.critical_issues + rule_results.error_issues +
                                 rule_results.warning_issues + rule_results.info_issues);
    } else {
        printf("Failed to check rule: %s\n", betanet_error_message(result));
    }

    printf("\n");

    // Generate SBOM
    printf("Generating SBOM...\n");
    betanet_Buffer sbom_json;
    result = linter_generate_sbom("../../", "spdx", &sbom_json);

    if (result == betanet_Success) {
        printf("SBOM generated successfully!\n");
        printf("SBOM size: %zu bytes\n", sbom_json.len);

        // Show first 200 characters of SBOM
        printf("SBOM preview:\n");
        for (size_t i = 0; i < 200 && i < sbom_json.len; i++) {
            printf("%c", sbom_json.data[i]);
        }
        if (sbom_json.len > 200) {
            printf("...\n");
        }
        printf("\n");

        betanet_buffer_free(sbom_json);
    } else {
        printf("Failed to generate SBOM: %s\n", betanet_error_message(result));
    }

    printf("\n");

    // Generate CycloneDX SBOM
    printf("Generating CycloneDX SBOM...\n");
    result = linter_generate_sbom("../../", "cyclonedx", &sbom_json);

    if (result == betanet_Success) {
        printf("CycloneDX SBOM generated successfully!\n");
        printf("SBOM size: %zu bytes\n", sbom_json.len);
        betanet_buffer_free(sbom_json);
    } else {
        printf("Failed to generate CycloneDX SBOM: %s\n", betanet_error_message(result));
    }

    // Cleanup
    linter_free(linter);
    betanet_cleanup();

    printf("\n=== Linter Example Complete ===\n");
    return 0;
}
