# TATA Motors Material Consumption Analyzer

## Live Application

**Test the application:** [https://tata-motors-app.onrender.com](https://tata-motors-app.onrender.com)

**Note:** Currently, only **Hassanpur depot** is operational due to time and data constraints during development. This serves as a proof-of-concept for the full multi-depot implementation.

## Overview

An intelligent web-based inventory management system that automates material consumption analysis and procurement processes for TATA Motors depots. The system transforms manual inventory processes into intelligent automated workflows with predictive analytics and cost optimization capabilities.

## Key Features

### Intelligent Material Classification
- **6-Tier Classification System:**
  - `HD_Critical`: High-demand, business-critical materials
  - `HD_Variable`: High-demand materials with variable consumption
  - `MD_Regular`: Medium-demand materials with consistent usage
  - `MD_Seasonal`: Medium-demand materials with seasonal variations
  - `LD_Stable`: Low-demand materials with stable consumption
  - `LD_Sporadic`: Low-demand materials with irregular patterns

### Advanced Inventory Analytics
- **4-Tier Status Assessment:**
  - `Critical`: Immediate ordering required
  - `Low`: Ordering recommended soon
  - `Adequate`: Sufficient stock levels
  - `Excess`: Cost optimization opportunities

### Predictive Analytics Engine
- Monthly coverage calculations based on historical patterns
- Optimal order quantity recommendations
- Consumption variability analysis
- Lead time and safety stock considerations
- Seasonal variation adjustments

### Smart Cart & Ordering System
- Automated cart population for critical/low stock items
- Manual material addition with search functionality
- Real-time price calculations
- Vendor-specific order organization

### Automated Vendor Management
- Intelligent vendor categorization (10+ vendor types)
- Consolidated vendor-specific order views
- Bulk order confirmation capabilities
- Scalable vendor framework

### Business Intelligence
- Inventory efficiency scoring
- Cost optimization analysis
- Material classification summaries
- Comprehensive reporting dashboard

## Technology Stack

- **Backend:** Python with Flask framework
- **Frontend:** HTML5, CSS3, JavaScript with Jinja2 templating
- **Data Processing:** Python analytics libraries
- **Database:** Supabase (cloud-hosted)
- **Deployment:** Render.com
- **Architecture:** Modular, scalable design

## Business Impact

### Operational Efficiency
- **Automated Analysis:** Eliminates time-consuming manual inventory processes
- **Predictive Ordering:** Prevents stockouts and reduces emergency orders
- **Vendor Coordination:** Streamlines multi-vendor management

### Cost Optimization
- **Inventory Reduction:** Identifies excess stock for immediate cost savings
- **Economic Ordering:** Calculates optimal order quantities
- **Vendor Efficiency:** Enables better vendor negotiations through consolidated orders

### Decision Support
- **Data-Driven Insights:** Scientific basis for all inventory decisions
- **Risk Management:** Early identification of critical material shortages
- **Performance Monitoring:** Continuous inventory efficiency tracking

## Current Limitations

- **Single Depot:** Only Hassanpur depot is currently operational
- **Data Scope:** Limited to available historical consumption data
- **Basic Authentication:** No role-based access control implemented

## Future Roadmap

### Phase 1: Authentication & Access Control
- [ ] **Role-Based Authentication:** Different access levels for various job roles
- [ ] **User Management:** Admin panels for user creation and permission management
- [ ] **Audit Trails:** Complete logging of all system interactions

### Phase 2: Vendor Integration
- [ ] **Direct Vendor Linking:** Real-time integration with vendor systems
- [ ] **Automated Purchase Orders:** Direct PO generation and transmission
- [ ] **Three-Way Matching:** Automated PO, receipt, and invoice reconciliation

### Phase 3: Global Expansion
- [ ] **Multi-Language Support:** Localization for different national depots
- [ ] **Currency Management:** Multi-currency support for international operations
- [ ] **Regional Compliance:** Adapting to local regulations and practices

### Phase 4: Financial Controls
- [ ] **Budget vs Actual Tracking:** Real-time spending monitoring
- [ ] **Budget Alerts:** Automated notifications for budget overruns
- [ ] **Financial Reporting:** Comprehensive cost analysis and reporting

### Phase 5: Offline Capabilities
- [ ] **Offline Mode:** Full functionality for remote locations
- [ ] **Data Synchronization:** Seamless sync when connectivity is restored
- [ ] **Progressive Web App:** Mobile-first offline experience

### Phase 6: Cloud Optimization
- [ ] **Enhanced Cloud Services:** Better performance and reliability
- [ ] **Auto-Scaling:** Dynamic resource allocation based on usage
- [ ] **Disaster Recovery:** Robust backup and recovery systems
- [ ] **API Gateway:** Standardized API access for integrations

## Technical Achievements

### Algorithm Development
- **Sophisticated Classification:** 6-tier material categorization system
- **Predictive Models:** Time-series forecasting with seasonal adjustments
- **Optimization Engines:** Economic order quantity calculations

### User Experience
- **Intuitive Design:** Complex analytics through user-friendly interfaces
- **Real-Time Updates:** Live pricing and inventory status
- **Progressive Enhancement:** Works across all device types

### Scalability
- **Modular Architecture:** Easy expansion and maintenance
- **Cloud-Ready:** Optimized for modern cloud deployments
- **Multi-Tenant:** Framework for multiple depot support

## Performance Metrics

- **7 Specialized Pages** for different business functions
- **10+ Vendor Categories** with expansion capability
- **6 Material Classifications** for comprehensive categorization
- **4-Tier Status System** for inventory management
- **Multi-Platform Deployment** ready configuration

## Contributing

This project was developed as part of a comprehensive inventory management solution for TATA Motors. For contributions or questions, please contact:

**Developer:** Abhimanyu Choudhry  
**Email:** choudhryabhimanyu31@gmail.com  

## License

This project is developed for TATA Motors internal use. All rights reserved.

## Acknowledgments

- **Project Guide:** Ms. Gayatri Bansal
- **Organization:** TATA Motors
- **Focus:** Hassanpur Depot Operations
