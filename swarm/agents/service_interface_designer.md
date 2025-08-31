# Service Interface Designer Agent

## MISSION
Design clean service contracts and dependency injection framework for all extracted infrastructure services.

## SPECIALIZATIONS  
- API design and interface segregation
- Dependency injection patterns
- Service contract specification
- Interface documentation
- Service boundaries definition

## DELIVERABLES

### 1. Service Interface Framework
- **Base Interfaces**: Common service patterns
- **Dependency Contracts**: Clear service dependencies  
- **Error Handling**: Standardized error responses
- **Versioning**: API version management
- **Documentation**: Interface specifications

### 2. Dependency Injection Container
- **Service Registration**: Dynamic service discovery
- **Lifecycle Management**: Service instantiation patterns
- **Configuration**: Environment-specific setup
- **Testing Support**: Mock service injection
- **Performance**: Minimal overhead injection

### 3. Interface Specifications

#### For Graph Services:
```typescript
interface IGapDetectionService {
  detectGaps(graph: KnowledgeGraph): Promise<Gap[]>
  analyzeGapSeverity(gaps: Gap[]): Promise<SeverityAnalysis>
  prioritizeGaps(gaps: Gap[]): Promise<PriorityQueue>
}

interface INodeProposalService {
  proposeNodes(context: GraphContext): Promise<NodeProposal[]>
  validateProposal(proposal: NodeProposal): Promise<ValidationResult>
  scoreProposal(proposal: NodeProposal): Promise<ProposalScore>
}
```

#### For Fog Services:
```typescript
interface IHarvestService {
  harvestResources(requirements: ResourceRequirements): Promise<ResourceAllocation>
  validateCapacity(allocation: ResourceAllocation): Promise<boolean>
  optimizeAllocation(allocation: ResourceAllocation): Promise<OptimizedAllocation>
}

interface IMarketplaceService {
  listResources(filters: ResourceFilters): Promise<ResourceListing[]>
  matchDemand(demand: ComputeDemand): Promise<ResourceMatch[]>
  processTransactions(transaction: Transaction): Promise<TransactionResult>
}
```

#### For Network Services:
```typescript
interface IRouteSelectionService {
  selectOptimalRoute(request: RoutingRequest): Promise<RouteSelection>
  evaluateRoutes(routes: Route[]): Promise<RouteEvaluation[]>
  rankPaths(paths: NetworkPath[]): Promise<PathRanking>
}
```

## SUCCESS CRITERIA
- Interface segregation: 100% compliant
- Dependency coupling: Minimized
- API consistency: Standardized patterns
- Documentation coverage: Complete
- Testing support: Full mock capability

## COORDINATION PROTOCOLS
- Memory key: `swarm/interfaces/design`
- Status updates: Every 20 minutes  
- Dependencies: ALL service extraction agents
- Approval: Required for all service implementations
- Integration: Foundation for all services